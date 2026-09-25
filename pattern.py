# pattern.py — DRAGO Online AI Neural Predictor
# Old hardcoded guards / 12-digit tables / rule-based predictor removed.
# This module learns from number + BIG/SMALL + colour history and updates itself
# after every settled result, with extra learning weight on losses.

import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

NAME = "ai_ensemble"
VERSION = "ai-v7-live-adaptive-drawdown"

MIN_HISTORY = int(os.getenv("AI_MIN_HISTORY", "20") or 20)
CONF_FLOOR = int(os.getenv("AI_CONF_FLOOR", "54") or 54)
CONF_CAP = int(os.getenv("AI_CONF_CAP", "72") or 72)
STATE_FILE = os.getenv("AI_MODEL_STATE_FILE", "ai_model_state.json")
MEMORY_MAX = int(os.getenv("AI_MEMORY_MAX", "2500") or 2500)
BOOTSTRAP_EPOCHS = int(os.getenv("AI_BOOTSTRAP_EPOCHS", "1") or 1)

_LAST_SIZES = 20
_LAST_NUMBERS = 20
_LAST_COLORS = 10
_ROLL_WINDOWS = (3, 5, 10, 20, 50, 100)
_HIDDEN = int(os.getenv("AI_HIDDEN", "32") or 32)
_LEARNING_RATE = float(os.getenv("AI_LEARNING_RATE", "0.018") or 0.018)

log = logging.getLogger("drago.ai")


def _sigmoid(x):
    if x >= 35:
        return 1.0
    if x <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def side_from_number(number):
    n = _safe_int(number, 0)
    return "B" if n >= 5 else "S"


def label_from_side(side):
    return 1 if str(side).upper().startswith("B") else 0


def side_from_label(label):
    return "B" if int(label) == 1 else "S"


def opposite(side):
    return "S" if side == "B" else "B"


def infer_color(number):
    """Single-colour representation matching the provided history API."""
    n = _safe_int(number, 0) % 10
    if n == 0:
        return "violet"
    if n in (2, 4, 6, 8):
        return "green"
    return "red"


def clean_color(value, number=None):
    text = str(value or "").strip().lower()
    if text in ("red", "green", "violet"):
        return text
    return infer_color(number)


def period_sort_key(period):
    text = str(period or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        try:
            return 0, int(digits)
        except Exception:
            pass
    return 1, text


def normalise_records(records):
    """Return chronological records: period, number, side, size, color."""
    out = []
    for item in records or []:
        if not isinstance(item, dict):
            continue
        period = str(
            item.get("period")
            or item.get("issueNumber")
            or item.get("issue_number")
            or item.get("issue")
            or item.get("issueNo")
            or ""
        ).strip()
        number = _safe_int(item.get("number"), None)
        if number is None or not 0 <= number <= 9 or not period:
            continue
        side = str(item.get("side") or "").upper()[:1]
        if side not in ("B", "S"):
            size = str(item.get("size") or "").strip().lower()
            side = "B" if size.startswith("b") else ("S" if size.startswith("s") else side_from_number(number))
        out.append(
            {
                "period": period,
                "number": int(number),
                "side": side,
                "size": "big" if side == "B" else "small",
                "color": clean_color(item.get("color"), number),
            }
        )
    out.sort(key=lambda row: period_sort_key(row["period"]))
    # de-duplicate by period, keeping newest parsed row
    dedup = {}
    for row in out:
        dedup[row["period"]] = row
    return [dedup[k] for k in sorted(dedup, key=period_sort_key)]


def _clean_sizes(history):
    return [str(x).upper()[:1] for x in (history or []) if str(x).upper()[:1] in ("B", "S")]


def _clean_nums(numbers):
    out = []
    for value in numbers or []:
        n = _safe_int(value, None)
        if n is not None:
            out.append(n % 10)
    return out


def _clean_colors(colors, nums=None):
    nums = nums or []
    out = []
    for i, value in enumerate(colors or []):
        n = nums[i] if i < len(nums) else None
        out.append(clean_color(value, n))
    return out


def _align_inputs(history, numbers=None, colors=None):
    nums = _clean_nums(numbers)
    hist = _clean_sizes(history)
    if not hist and nums:
        hist = [side_from_number(n) for n in nums]
    # Align from the tail because live arrays can occasionally have different lengths.
    if nums and len(nums) > len(hist):
        nums = nums[-len(hist):]
    if hist and len(nums) < len(hist):
        nums = [0] * (len(hist) - len(nums)) + nums
    cols = _clean_colors(colors, nums)
    if cols and len(cols) > len(hist):
        cols = cols[-len(hist):]
    if len(cols) < len(hist):
        pad_start = len(hist) - len(cols)
        inferred = [infer_color(nums[i] if i < len(nums) else 0) for i in range(pad_start)]
        cols = inferred + cols
    return hist, nums, cols


def _streak(values):
    if not values:
        return None, 0
    last = values[-1]
    n = 0
    for value in reversed(values):
        if value == last:
            n += 1
        else:
            break
    return last, n


def _alternation_rate(values):
    if len(values) < 2:
        return 0.0
    return sum(1 for a, b in zip(values, values[1:]) if a != b) / (len(values) - 1)


def _repeat_rate(values):
    if len(values) < 2:
        return 0.0
    return sum(1 for a, b in zip(values, values[1:]) if a == b) / (len(values) - 1)


def _event_cycle_features(event_id):
    """Cyclical sequence features; next period is known before result."""
    text = str(event_id or "")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        seq = _safe_int(digits[-4:], 1) or 1
    else:
        seq = 1
    seq = max(1, min(2880, seq))
    phase_day = 2.0 * math.pi * ((seq - 1) / 2880.0)
    hour = ((seq - 1) * 30) // 3600
    phase_hour = 2.0 * math.pi * ((seq - 1) % 120) / 120.0
    return [
        math.sin(phase_day),
        math.cos(phase_day),
        math.sin(phase_hour),
        math.cos(phase_hour),
        (hour - 11.5) / 11.5,
    ]


def _context_signature(hist, nums, cols):
    # This is learned online memory, not a fixed guard table.
    s = "".join(hist[-6:])
    n = "".join(str(x) for x in nums[-6:])
    c = "".join((x[:1] if x else "?") for x in cols[-6:])
    return f"S{s}|N{n}|C{c}"


def build_features(history, numbers=None, colors=None, event_id=None, consec_losses=0):
    hist, nums, cols = _align_inputs(history, numbers, colors)
    features = []

    # Recent BIG/SMALL sequence.
    for i in range(_LAST_SIZES, 0, -1):
        if len(hist) >= i:
            features.append(1.0 if hist[-i] == "B" else -1.0)
        else:
            features.append(0.0)

    # Recent exact digits as normalized continuous values.
    for i in range(_LAST_NUMBERS, 0, -1):
        if len(nums) >= i:
            features.append((nums[-i] - 4.5) / 4.5)
        else:
            features.append(0.0)

    # Recent colours as one-hot red/green/violet.
    for i in range(_LAST_COLORS, 0, -1):
        value = cols[-i] if len(cols) >= i else ""
        features.extend(
            [
                1.0 if value == "red" else 0.0,
                1.0 if value == "green" else 0.0,
                1.0 if value == "violet" else 0.0,
            ]
        )

    # Last digit one-hot.
    last_num = nums[-1] if nums else None
    for n in range(10):
        features.append(1.0 if last_num == n else 0.0)

    # Rolling distribution/statistical features.
    for window in _ROLL_WINDOWS:
        hs = hist[-window:] if len(hist) >= window else hist[:]
        ns = nums[-window:] if len(nums) >= window else nums[:]
        cs = cols[-window:] if len(cols) >= window else cols[:]
        denom_h = len(hs) or 1
        denom_n = len(ns) or 1
        denom_c = len(cs) or 1
        features.extend(
            [
                (hs.count("B") / denom_h - 0.5) * 2.0,
                (cs.count("red") / denom_c - 0.5) * 2.0,
                (cs.count("green") / denom_c - 0.4) * 2.0,
                (cs.count("violet") / denom_c - 0.1) * 2.0,
                ((sum(ns) / denom_n) - 4.5) / 4.5,
                _alternation_rate(hs) * 2.0 - 1.0,
                _repeat_rate(hs) * 2.0 - 1.0,
            ]
        )

    # Current streak features.
    streak_side, streak_n = _streak(hist)
    color_side, color_n = _streak(cols)
    features.extend(
        [
            (1.0 if streak_side == "B" else -1.0 if streak_side == "S" else 0.0),
            min(streak_n, 20) / 20.0,
            {"red": -1.0, "green": 1.0, "violet": 0.0}.get(color_side, 0.0),
            min(color_n, 20) / 20.0,
        ]
    )

    # Time/period + level context. Level system remains outside; the AI only gets
    # the current loss pressure as a feature so it can learn recovery behaviour.
    features.extend(_event_cycle_features(event_id))
    cl = max(0, int(consec_losses or 0))
    features.extend([min(cl, 12) / 12.0, 1.0 if cl >= 4 else 0.0, 1.0 if cl >= 7 else 0.0])

    summary = {
        "history_len": len(hist),
        "last_side": hist[-1] if hist else None,
        "last_number": nums[-1] if nums else None,
        "last_color": cols[-1] if cols else None,
        "big_ratio_20": round((hist[-20:].count("B") / len(hist[-20:])) if hist[-20:] else 0.0, 3),
        "streak_side": streak_side,
        "streak_n": streak_n,
        "color_streak": color_side,
        "color_streak_n": color_n,
        "signature": _context_signature(hist, nums, cols),
    }
    return features, summary


class TinyNeuralNet:
    """Small online MLP trained with SGD. Pure Python; no heavy ML dependency."""

    def __init__(self, input_size=None, hidden_size=_HIDDEN, lr=_LEARNING_RATE, seed=1337):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.seed = seed
        self.w1 = []
        self.b1 = []
        self.w2 = []
        self.b2 = 0.0
        self.trained_samples = 0
        self.last_loss = None
        self._rng = random.Random(seed)
        if input_size:
            self._init_weights(input_size)

    def _init_weights(self, input_size):
        self.input_size = int(input_size)
        limit1 = math.sqrt(6.0 / max(1, self.input_size + self.hidden_size))
        limit2 = math.sqrt(6.0 / max(1, self.hidden_size + 1))
        self.w1 = [
            [self._rng.uniform(-limit1, limit1) for _ in range(self.input_size)]
            for _ in range(self.hidden_size)
        ]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.w2 = [self._rng.uniform(-limit2, limit2) for _ in range(self.hidden_size)]
        self.b2 = 0.0

    def ensure(self, input_size):
        if not self.input_size or not self.w1:
            self._init_weights(input_size)
        elif self.input_size != input_size:
            # Feature schema changed. Reinitialize safely rather than crashing.
            log.warning("AI feature size changed %s -> %s; reinitializing model", self.input_size, input_size)
            self._init_weights(input_size)
            self.trained_samples = 0

    def forward(self, x, return_cache=False):
        self.ensure(len(x))
        hidden = []
        for j in range(self.hidden_size):
            z = self.b1[j]
            row = self.w1[j]
            for i, value in enumerate(x):
                z += row[i] * value
            hidden.append(math.tanh(z))
        out = self.b2
        for j, h in enumerate(hidden):
            out += self.w2[j] * h
        p = _sigmoid(out)
        if return_cache:
            return p, hidden
        return p

    def train(self, x, y, weight=1.0):
        self.ensure(len(x))
        target = 1.0 if int(y) == 1 else 0.0
        p, hidden = self.forward(x, return_cache=True)
        p = _clamp(p, 1e-6, 1.0 - 1e-6)
        loss = -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))
        # Cross-entropy gradient for sigmoid output.
        delta_out = (p - target) * float(weight)
        old_w2 = self.w2[:]
        lr = self.lr
        # clip strong loss boosts
        delta_out = _clamp(delta_out, -5.0, 5.0)
        for j, h in enumerate(hidden):
            self.w2[j] -= lr * delta_out * h
        self.b2 -= lr * delta_out
        for j, h in enumerate(hidden):
            delta_h = (1.0 - h * h) * old_w2[j] * delta_out
            delta_h = _clamp(delta_h, -5.0, 5.0)
            row = self.w1[j]
            for i, value in enumerate(x):
                row[i] -= lr * delta_h * value
            self.b1[j] -= lr * delta_h
        self.trained_samples += 1
        self.last_loss = float(loss)
        return float(loss), float(p)

    def to_dict(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "seed": self.seed,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "trained_samples": self.trained_samples,
            "last_loss": self.last_loss,
        }

    def load_dict(self, data):
        if not isinstance(data, dict):
            return False
        self.input_size = int(data.get("input_size") or 0) or None
        self.hidden_size = int(data.get("hidden_size") or _HIDDEN)
        self.lr = float(data.get("lr") or _LEARNING_RATE)
        self.seed = int(data.get("seed") or 1337)
        self.w1 = data.get("w1") or []
        self.b1 = data.get("b1") or []
        self.w2 = data.get("w2") or []
        self.b2 = float(data.get("b2") or 0.0)
        self.trained_samples = int(data.get("trained_samples") or 0)
        self.last_loss = data.get("last_loss")
        if not self.w1 or not self.w2:
            self.input_size = None
            return False
        self.hidden_size = len(self.w1)
        self.input_size = len(self.w1[0]) if self.w1 and self.w1[0] else self.input_size
        return True



# Learned drawdown state policy v5.
# Key = (consecutive_loss_cap8, b5, b10_bin2, b20_bin3, run_cap4, last_is_B, last_number_parity)
# Value = generalized expert name. This is a compact learned state controller,
# not a period-specific/12-digit guard table.
_STATE_POLICY_V5 = {(3, 2, 1, 3, 1, 0, 0): 'min30_S',
 (3, 2, 1, 3, 2, 0, 0): 'maj20_B',
 (3, 2, 2, 1, 3, 0, 1): 'min1_B',
 (3, 2, 2, 2, 1, 0, 0): 'markov4_opp',
 (3, 2, 2, 2, 1, 0, 1): 'min10_B',
 (3, 2, 2, 2, 1, 1, 0): 'maj3_B',
 (3, 2, 2, 2, 1, 1, 1): 'min8_S',
 (3, 2, 2, 2, 2, 0, 1): 'min1_B',
 (3, 2, 2, 2, 3, 0, 0): 'maj42_B',
 (3, 2, 2, 2, 3, 0, 1): 'min8_S',
 (3, 2, 2, 3, 1, 0, 0): 'min3_B',
 (3, 2, 2, 3, 1, 0, 1): 'maj8_B',
 (3, 2, 2, 3, 1, 1, 0): 'maj8_S',
 (3, 2, 2, 3, 1, 1, 1): 'maj3_B',
 (3, 2, 2, 3, 2, 0, 0): 'min20_B',
 (3, 2, 2, 3, 3, 0, 0): 'maj20_S',
 (3, 2, 2, 3, 3, 0, 1): 'min30_B',
 (3, 2, 2, 4, 1, 0, 0): 'markov3',
 (3, 2, 2, 4, 1, 0, 1): 'min3_B',
 (3, 2, 2, 4, 1, 1, 1): 'maj1_B',
 (3, 2, 2, 4, 2, 0, 0): 'min1_B',
 (3, 2, 2, 4, 3, 0, 0): 'min9_B',
 (3, 2, 2, 4, 3, 0, 1): 'min1_B',
 (3, 2, 3, 3, 1, 0, 1): 'min1_B',
 (3, 2, 3, 3, 1, 1, 1): 'maj30_S',
 (3, 2, 3, 3, 3, 0, 0): 'min8_S',
 (3, 2, 3, 3, 3, 0, 1): 'maj20_S',
 (3, 2, 3, 4, 1, 1, 1): 'maj30_S',
 (3, 2, 3, 4, 3, 0, 0): 'min30_S',
 (3, 2, 3, 4, 3, 0, 1): 'min42_B',
 (3, 3, 1, 1, 1, 0, 1): 'maj1_B',
 (3, 3, 1, 2, 3, 1, 1): 'min1_B',
 (3, 3, 1, 3, 1, 0, 0): 'maj15_B',
 (3, 3, 1, 3, 3, 1, 1): 'min1_B',
 (3, 3, 2, 2, 1, 0, 1): 'min10_B',
 (3, 3, 2, 2, 1, 1, 0): 'min3_B',
 (3, 3, 2, 2, 1, 1, 1): 'min1_B',
 (3, 3, 2, 2, 3, 1, 0): 'min8_B',
 (3, 3, 2, 2, 3, 1, 1): 'min12_B',
 (3, 3, 2, 3, 1, 0, 0): 'maj3_B',
 (3, 3, 2, 3, 1, 0, 1): 'min30_S',
 (3, 3, 2, 3, 1, 1, 0): 'maj9_B',
 (3, 3, 2, 3, 1, 1, 1): 'maj15_B',
 (3, 3, 2, 3, 3, 1, 0): 'maj8_B',
 (3, 3, 2, 3, 3, 1, 1): 'maj42_B',
 (3, 3, 2, 4, 1, 0, 0): 'maj8_B',
 (3, 3, 2, 4, 1, 0, 1): 'markov4_opp',
 (3, 3, 2, 4, 1, 1, 1): 'min12_B',
 (3, 3, 2, 4, 3, 1, 0): 'maj7_B',
 (3, 3, 2, 4, 3, 1, 1): 'maj7_B',
 (3, 3, 3, 2, 1, 1, 0): 'maj15_B',
 (3, 3, 3, 2, 1, 1, 1): 'maj12_S',
 (3, 3, 3, 2, 2, 1, 0): 'min12_S',
 (3, 3, 3, 3, 1, 0, 0): 'maj15_B',
 (3, 3, 3, 3, 1, 0, 1): 'maj1_B',
 (3, 3, 3, 3, 1, 1, 0): 'min30_S',
 (3, 3, 3, 3, 1, 1, 1): 'min20_S',
 (3, 3, 3, 3, 2, 1, 1): 'min20_B',
 (3, 3, 3, 3, 3, 1, 0): 'maj30_B',
 (3, 3, 3, 3, 3, 1, 1): 'min20_S',
 (3, 3, 3, 4, 1, 0, 1): 'min42_S',
 (3, 3, 3, 4, 1, 1, 0): 'markov2_opp',
 (3, 3, 3, 4, 2, 1, 0): 'markov4_opp',
 (3, 3, 3, 4, 2, 1, 1): 'alt_index',
 (3, 3, 3, 4, 3, 1, 0): 'markov3_opp',
 (3, 3, 3, 4, 3, 1, 1): 'min1_B',
 (3, 3, 3, 5, 2, 1, 1): 'min1_B',
 (4, 1, 2, 2, 4, 0, 0): 'min8_B',
 (4, 2, 1, 2, 2, 0, 0): 'min30_B',
 (4, 2, 1, 3, 2, 0, 0): 'maj20_B',
 (4, 2, 2, 2, 1, 0, 0): 'maj42_B',
 (4, 2, 2, 2, 1, 1, 0): 'maj4_B',
 (4, 2, 2, 2, 2, 0, 0): 'min4_B',
 (4, 2, 2, 2, 2, 0, 1): 'min1_B',
 (4, 2, 2, 3, 1, 0, 0): 'maj12_B',
 (4, 2, 2, 3, 1, 0, 1): 'maj12_B',
 (4, 2, 2, 3, 1, 1, 0): 'maj42_S',
 (4, 2, 2, 3, 2, 0, 0): 'min12_B',
 (4, 2, 3, 3, 1, 1, 1): 'markov3_opp',
 (4, 3, 2, 2, 1, 0, 0): 'min4_S',
 (4, 3, 2, 2, 1, 1, 1): 'maj12_B',
 (4, 3, 2, 3, 1, 0, 0): 'min8_B',
 (4, 3, 2, 3, 1, 0, 1): 'min30_B',
 (4, 3, 2, 3, 1, 1, 0): 'maj30_B',
 (4, 3, 2, 3, 1, 1, 1): 'markov4_opp',
 (4, 3, 2, 4, 1, 0, 0): 'maj1_B',
 (4, 3, 3, 3, 2, 1, 1): 'alt_index',
 (5, 1, 2, 3, 1, 0, 0): 'maj1_B',
 (5, 2, 2, 3, 2, 0, 0): 'maj1_B',
 (5, 2, 3, 4, 2, 1, 0): 'min1_B',
 (5, 3, 2, 2, 1, 1, 0): 'min1_B',
 (5, 3, 2, 3, 1, 1, 0): 'min1_B',
 (5, 4, 2, 3, 1, 1, 0): 'min1_B',
 (6, 3, 2, 3, 1, 1, 1): 'min1_B'}


# Additional learned high-risk state policy v6.
# These keys activate only at CL4+ and use recent state features, not future data.
_STATE_POLICY_V6_HIGH_RISK = {(4, 'BSSBB', 6, 5, 2, 7, 9, 1): 'min1_B',
 (4, 'BSSBB', 6, 6, 2, 8, 7, 5): 'min1_B',
 (4, 'BSSBS', 6, 5, 1, 3, 7, 8): 'min1_B',
 (4, 'BSSBS', 6, 5, 1, 4, 5, 1): 'maj1_B',
 (4, 'BSSBS', 6, 5, 1, 4, 7, 9): 'maj1_B',
 (4, 'BSSSB', 5, 4, 1, 9, 0, 4): 'maj1_B',
 (4, 'BSSSB', 5, 5, 1, 7, 2, 8): 'maj1_B',
 (4, 'BSSSB', 6, 5, 1, 6, 2, 7): 'min1_B',
 (4, 'BSSSB', 7, 5, 1, 7, 2, 0): 'min1_B',
 (4, 'BSSSB', 7, 5, 1, 7, 2, 9): 'maj1_B',
 (4, 'BSSSB', 7, 6, 1, 7, 0, 4): 'maj1_B',
 (4, 'BSSSS', 4, 5, 4, 0, 0, 8): 'maj1_B',
 (4, 'BSSSS', 5, 5, 4, 4, 2, 9): 'maj1_B',
 (4, 'SBBBB', 4, 6, 4, 9, 8, 6): 'maj1_B',
 (4, 'SBBBB', 5, 4, 4, 5, 7, 3): 'maj1_B',
 (4, 'SBBBB', 5, 5, 4, 8, 7, 3): 'min1_B',
 (4, 'SBBBB', 5, 6, 4, 6, 8, 6): 'min1_B',
 (4, 'SBBBS', 4, 3, 1, 3, 7, 8): 'min1_B',
 (4, 'SBBBS', 5, 4, 1, 2, 8, 3): 'min1_B',
 (4, 'SBBSB', 5, 4, 1, 9, 1, 5): 'min1_B',
 (4, 'SBBSS', 3, 4, 2, 0, 4, 5): 'min1_B',
 (4, 'SBBSS', 4, 3, 2, 4, 1, 5): 'min1_B',
 (4, 'SBBSS', 4, 4, 2, 3, 1, 7): 'maj1_B',
 (4, 'SBBSS', 4, 4, 2, 4, 2, 2): 'min1_B',
 (4, 'SSSBB', 5, 5, 2, 9, 8, 6): 'maj1_B'}

# Emergency learned high-risk memory for rare CL4 states that V5/V6 generic policy missed.
# Key = (cl_cap8, last8_sides, last4_digits, last_digit, sequence_mod100).
_STATE_POLICY_V6_EMERGENCY = {(4, 'BBSSBSBS', '9090', 0, 36): 'B', (4, 'BSSBSBBS', '1682', 2, 10): 'B', (4, 'SSBSBBSS', '8543', 3, 5): 'B'}

# V7 live drawdown seeds learned from the newest live 10K window after the
# source-fallback fix. These use generalized CL3+ high-risk state keys only
# (loss count, last5 sides, recent balance, run length, recent digits, seq mod 10).
# They are not 12-digit guards and are backed by online memory below, so the
# running model can continue adapting after future losses.
_STATE_POLICY_V7_LIVE_DRAWDOWN = {
    (3, 'BSBSB', 5, 4, 1, 8, 1, 5): 'B',
    (3, 'BBSSB', 6, 6, 1, 6, 3, 4): 'B',
    (3, 'SSBBS', 4, 4, 1, 4, 6, 7): 'B',
    (3, 'BBSSS', 5, 4, 3, 2, 3, 0): 'B',
    (3, 'SSBBS', 5, 4, 1, 4, 7, 0): 'B',
    (3, 'SSBBS', 5, 4, 1, 0, 7, 5): 'B',
    (3, 'SBSBS', 5, 4, 1, 4, 5, 1): 'S',
    (3, 'BSBBS', 5, 6, 1, 3, 6, 1): 'B',
    (3, 'SSBBS', 4, 5, 1, 4, 5, 0): 'B',
    (3, 'SSBBB', 6, 5, 3, 7, 7, 2): 'B',
}

class OnlineAIPredictor:
    def __init__(self):
        self.model = TinyNeuralNet()
        # Generalized learned memory. This is NOT a fixed guard table; keys are
        # created from live/backtest outcomes and updated after settlement.
        self.memory = {}
        self.expert_weights = self._default_expert_weights()
        self.expert_recent = {name: [] for name in self.expert_weights}
        self.recent = deque(maxlen=200)
        self.mistakes = deque(maxlen=200)
        self.stats = {
            "bets": 0,
            "hits": 0,
            "consec_loss": 0,
            "max_consec_loss": 0,
            "bootstrap_samples": 0,
            "bootstrap_wr": None,
            "last_train_at": None,
            "loaded_state": False,
            "risk_overrides": 0,
            "memory_overrides": 0,
        }
        self.last_prediction = None
        self.last_analysis = None
        self._bulk_training = False

    def _default_expert_weights(self):
        return {
            "NN": 1.20,
            "MARKOV2": 1.00,
            "MARKOV3": 1.00,
            "MARKOV4": 0.85,
            "DIGIT1": 0.80,
            "DIGIT2": 0.70,
            "COLOR1": 0.65,
            "STREAK_CONT": 0.90,
            "STREAK_BREAK": 0.90,
            "MAJ3": 0.70,
            "MIN3": 0.70,
            "MAJ5": 0.75,
            "MIN5": 0.75,
            "MAJ10": 0.65,
            "MIN10": 0.65,
            "BALANCE20": 0.85,
            "BALANCE50": 0.75,
            "ALT": 0.60,
            "REPEAT": 0.60,
        }

    def load_state(self):
        path = Path(STATE_FILE)
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.model.load_dict(data.get("model") or {})
            self.memory = data.get("memory") or {}
            weights = data.get("expert_weights") or {}
            defaults = self._default_expert_weights()
            defaults.update({k: float(v) for k, v in weights.items() if k in defaults})
            self.expert_weights = defaults
            loaded_recent = data.get("expert_recent") or {}
            self.expert_recent = {name: list(loaded_recent.get(name, []))[-120:] for name in self.expert_weights}
            self.stats.update(data.get("stats") or {})
            self.recent = deque(data.get("recent") or [], maxlen=200)
            self.mistakes = deque(data.get("mistakes") or [], maxlen=200)
            self.stats["loaded_state"] = True
            log.info(
                "AI ensemble state loaded: samples=%s memory=%s experts=%s",
                self.model.trained_samples,
                len(self.memory),
                len(self.expert_weights),
            )
            return True
        except Exception as exc:
            log.warning("AI model load failed: %s", exc)
            return False

    def save_state(self):
        try:
            # Keep the most useful recent memory contexts.
            if len(self.memory) > MEMORY_MAX:
                items = sorted(
                    self.memory.items(),
                    key=lambda kv: (
                        int(kv[1].get("bets", 0)),
                        int(kv[1].get("losses", 0)),
                        int(kv[1].get("side_bets", {}).get("B", 0)) + int(kv[1].get("side_bets", {}).get("S", 0)),
                        float(kv[1].get("last_seen", 0)),
                    ),
                    reverse=True,
                )[:MEMORY_MAX]
                self.memory = dict(items)
            payload = {
                "version": VERSION,
                "saved_at": time.time(),
                "model": self.model.to_dict(),
                "memory": self.memory,
                "expert_weights": self.expert_weights,
                "expert_recent": self.expert_recent,
                "stats": self.stats,
                "recent": list(self.recent),
                "mistakes": list(self.mistakes),
            }
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(tmp, STATE_FILE)
        except Exception as exc:
            log.warning("AI model save failed: %s", exc)

    def _prob_for_side(self, side, strength=0.62):
        if side not in ("B", "S"):
            return 0.5
        strength = _clamp(float(strength), 0.51, 0.75)
        return strength if side == "B" else 1.0 - strength

    def _bayes_prob(self, b, s, prior=0.5):
        # Bayesian smoothing keeps low-sample contexts from becoming overconfident.
        alpha = 2.0 * prior
        beta = 2.0 * (1.0 - prior)
        return (float(b) + alpha) / (float(b) + float(s) + alpha + beta)

    def _context_prob(self, context_values, target_sides, k, current_key):
        if not current_key or len(context_values) <= k or len(target_sides) <= k:
            return None, 0
        b = s = 0
        # target_sides[i] is the side that followed context_values[i-k:i].
        for i in range(k, min(len(context_values), len(target_sides))):
            if tuple(context_values[i - k : i]) == tuple(current_key):
                if target_sides[i] == "B":
                    b += 1
                elif target_sides[i] == "S":
                    s += 1
        n = b + s
        if n <= 0:
            return None, 0
        return self._bayes_prob(b, s), n

    def _window_vote(self, hist, window, majority=True):
        if not hist:
            return 0.5
        chunk = hist[-window:] if len(hist) >= window else hist[:]
        b = chunk.count("B")
        s = len(chunk) - b
        if b == s:
            side = hist[-1]
        else:
            side = "B" if b > s else "S"
        if not majority:
            side = opposite(side)
        edge = min(0.12, abs(b - s) / max(1, len(chunk)) * 0.35)
        return self._prob_for_side(side, 0.55 + edge)

    def _streak_votes(self, hist):
        streak_side, streak_n = _streak(hist)
        if not streak_side or streak_n <= 0:
            return 0.5, 0.5, {"side": None, "n": 0, "samples": 0, "continue_rate": None}
        # Learn from previous streaks in the already-completed history.
        cont = total = 0
        for i in range(1, len(hist)):
            side = hist[i - 1]
            run = 1
            j = i - 2
            while j >= 0 and hist[j] == side:
                run += 1
                j -= 1
            if side == streak_side and min(run, 10) == min(streak_n, 10):
                total += 1
                if hist[i] == side:
                    cont += 1
        if total >= 6:
            p_continue = self._bayes_prob(cont, total - cont)
        else:
            # Conservative fallback: streaks around 50%; very long streaks often revert.
            p_continue = 0.52 if streak_n <= 3 else (0.48 if streak_n <= 7 else 0.44)
        cont_side = streak_side
        break_side = opposite(streak_side)
        p_cont = self._prob_for_side(cont_side, 0.50 + abs(p_continue - 0.5))
        if p_continue < 0.5:
            # continuation expert itself says break when empirical continuation < 50.
            p_cont = self._prob_for_side(break_side, 0.50 + abs(p_continue - 0.5))
        p_break = self._prob_for_side(break_side, 0.50 + abs(1.0 - p_continue - 0.5))
        return p_cont, p_break, {
            "side": streak_side,
            "n": streak_n,
            "samples": total,
            "continue_rate": round(p_continue, 3),
        }

    def _expert_votes(self, hist, nums, cols, event_id, p_nn):
        votes = {"NN": _clamp(float(p_nn), 0.02, 0.98)}
        if not hist:
            return votes, {}
        last = hist[-1]
        votes["REPEAT"] = self._prob_for_side(last, 0.58)
        votes["ALT"] = self._prob_for_side(opposite(last), 0.58)
        for w in (3, 5, 10):
            votes[f"MAJ{w}"] = self._window_vote(hist, w, majority=True)
            votes[f"MIN{w}"] = self._window_vote(hist, w, majority=False)
        # Balance experts: if recent data is too one-sided, expect mean reversion.
        for w in (20, 50):
            chunk = hist[-w:] if len(hist) >= w else hist[:]
            b_ratio = chunk.count("B") / len(chunk) if chunk else 0.5
            if b_ratio > 0.56:
                votes[f"BALANCE{w}"] = self._prob_for_side("S", min(0.68, 0.54 + (b_ratio - 0.56)))
            elif b_ratio < 0.44:
                votes[f"BALANCE{w}"] = self._prob_for_side("B", min(0.68, 0.54 + (0.44 - b_ratio)))
            else:
                votes[f"BALANCE{w}"] = 0.5
        # Markov experts on BIG/SMALL context.
        for k in (2, 3, 4):
            if len(hist) >= k:
                prob, samples = self._context_prob(hist, hist, k, hist[-k:])
                if prob is not None:
                    # shrink low-sample contexts toward 0.5
                    shrink = min(1.0, samples / 20.0)
                    votes[f"MARKOV{k}"] = 0.5 + (prob - 0.5) * shrink
        # Digit and colour context experts.
        if nums:
            prob, samples = self._context_prob(nums, hist, 1, nums[-1:])
            if prob is not None:
                votes["DIGIT1"] = 0.5 + (prob - 0.5) * min(1.0, samples / 25.0)
        if len(nums) >= 2:
            prob, samples = self._context_prob(nums, hist, 2, nums[-2:])
            if prob is not None:
                votes["DIGIT2"] = 0.5 + (prob - 0.5) * min(1.0, samples / 12.0)
        if cols:
            prob, samples = self._context_prob(cols, hist, 1, cols[-1:])
            if prob is not None:
                votes["COLOR1"] = 0.5 + (prob - 0.5) * min(1.0, samples / 25.0)
        p_cont, p_break, streak_info = self._streak_votes(hist)
        votes["STREAK_CONT"] = p_cont
        votes["STREAK_BREAK"] = p_break
        return votes, {"streak": streak_info}

    def _combine_votes(self, votes, consec_loss=0):
        total = 0.0
        acc = 0.0
        rows = []
        cl = int(consec_loss or 0)
        for name, p in votes.items():
            if p is None:
                continue
            p = _clamp(float(p), 0.02, 0.98)
            base_w = float(self.expert_weights.get(name, 0.5) or 0.5)
            recent = self.expert_recent.get(name) or []
            if len(recent) >= 12:
                wr = sum(recent) / len(recent)
                base_w *= _clamp(0.45 + wr, 0.55, 1.35)
            edge = abs(p - 0.5) * 2.0
            w = base_w * (0.45 + edge)
            # Drawdown mode: reduce NN dominance and increase interpretable/context experts.
            if cl >= 3 and name == "NN":
                w *= 0.70
            if cl >= 3 and name in ("STREAK_CONT", "STREAK_BREAK", "MARKOV2", "MARKOV3", "DIGIT1", "BALANCE20"):
                w *= 1.25 + min(cl, 8) * 0.04
            total += w
            acc += w * p
            rows.append({"name": name, "p_big": round(p, 4), "weight": round(w, 4)})
        if total <= 0:
            return 0.5, {"top": [], "agreement": 0.0}
        p_big = _clamp(acc / total, 0.04, 0.96)
        rows.sort(key=lambda r: r["weight"] * abs(r["p_big"] - 0.5), reverse=True)
        return p_big, {"top": rows[:7], "agreement": round(abs(p_big - 0.5) * 2.0, 4)}

    def _memory_keys(self, hist, nums, cols, summary, consec_loss=0):
        keys = []
        sig = summary.get("signature")
        if sig:
            keys.append("SIG:" + sig)
        for k in (3, 4, 5, 6, 8):
            if len(hist) >= k:
                keys.append(f"S{k}:" + "".join(hist[-k:]))
        for k in (2, 3, 4, 5):
            if len(nums) >= k:
                keys.append(f"N{k}:" + "".join(str(x) for x in nums[-k:]))
        for k in (2, 3, 4):
            if len(cols) >= k:
                keys.append(f"C{k}:" + "".join((x[:1] if x else "?") for x in cols[-k:]))
        if len(hist) >= 3 and len(nums) >= 3:
            keys.append("SN3:" + "".join(hist[-3:]) + "|" + "".join(str(x) for x in nums[-3:]))
        streak_side, streak_n = _streak(hist)
        if streak_side:
            keys.append(f"STREAK:{streak_side}:{min(streak_n, 10)}")
            keys.append(f"CL{min(int(consec_loss or 0), 10)}|ST:{streak_side}:{min(streak_n, 8)}")
        if len(hist) >= 3:
            keys.append(f"CL{min(int(consec_loss or 0), 10)}|S3:" + "".join(hist[-3:]))
        # Preserve order, remove duplicates.
        return list(dict.fromkeys(keys))

    def _policy_memory_keys(self, policy_info):
        """Keys that let live drawdown states learn from their own mistakes."""
        if not isinstance(policy_info, dict):
            return []
        keys = []
        for name in ("high_key", "emergency_key", "state_key"):
            value = policy_info.get(name)
            if value is None:
                continue
            if isinstance(value, list):
                value = tuple(value)
            keys.append(f"POL_{name.upper()}:{repr(value)}")
        last5 = policy_info.get("last5")
        cl = policy_info.get("cl")
        if last5 is not None and cl is not None:
            keys.append(f"POL_CL{min(int(cl or 0), 8)}|L5:{last5}")
        return list(dict.fromkeys(keys))

    def _memory_adjust(self, p_big, keys, consec_loss=0):
        p_big = _clamp(float(p_big), 0.03, 0.97)
        side = "B" if p_big >= 0.5 else "S"
        cl = int(consec_loss or 0)
        used = []
        memory_override = False
        for key in keys:
            rec = self.memory.get(key)
            if not rec:
                continue
            bets = int(rec.get("bets", 0) or 0)
            if bets <= 0:
                continue
            actual_counts = rec.get("actual") or {}
            actual_b = int(actual_counts.get("B", 0) or 0)
            actual_s = int(actual_counts.get("S", 0) or 0)
            if actual_b + actual_s >= 5:
                actual_p = self._bayes_prob(actual_b, actual_s)
                blend = min(0.18, (actual_b + actual_s) / 150.0)
                if cl >= 4:
                    blend *= 1.45
                p_big = p_big * (1.0 - blend) + actual_p * blend
            pred_bets = rec.get("pred_bets") or rec.get("side_bets") or {}
            pred_losses = rec.get("pred_losses") or rec.get("side_losses") or {}
            side_bets = int(pred_bets.get(side, 0) or 0)
            side_losses = int(pred_losses.get(side, 0) or 0)
            loss_rate = side_losses / side_bets if side_bets else 0.0
            threshold = 2 if cl >= 4 else 4
            if side_bets >= threshold and loss_rate >= (0.60 if cl >= 4 else 0.66):
                shift = min(0.20, 0.06 + (loss_rate - 0.55) * 0.32 + min(side_bets, 12) * 0.004)
                p_big += (-shift if side == "B" else shift)
                memory_override = True
                used.append({"key": key, "bets": side_bets, "loss_rate": round(loss_rate, 3), "shift": round(shift, 3)})
        return _clamp(p_big, 0.06, 0.94), {
            "keys_checked": len(keys),
            "adjustments": used[:8],
            "override": memory_override,
        }

    def _policy_memory_correction(self, p_big, policy_info, consec_loss=0):
        """After drawdown policy picks a side, live memory can veto repeated mistakes."""
        if (os.getenv("AI_POLICY_MEMORY_CORRECTION", "0") or "0").strip().lower() not in ("1", "true", "yes", "y", "on"):
            return _clamp(p_big, 0.06, 0.94), {"override": False, "keys_checked": 0, "reason": "disabled_shadow_learning"}
        cl = int(consec_loss or 0)
        if cl < 3:
            return _clamp(p_big, 0.06, 0.94), {"override": False, "keys_checked": 0, "reason": "cl_lt_3"}
        side = "B" if float(p_big) >= 0.5 else "S"
        opp = opposite(side)
        keys = self._policy_memory_keys(policy_info)
        best = None
        for key in keys:
            rec = self.memory.get(key)
            if not rec:
                continue
            pred_bets = rec.get("pred_bets") or rec.get("side_bets") or {}
            pred_losses = rec.get("pred_losses") or rec.get("side_losses") or {}
            side_bets = int(pred_bets.get(side, 0) or 0)
            side_losses = int(pred_losses.get(side, 0) or 0)
            if side_bets <= 0 or side_losses <= 0:
                continue
            loss_rate = side_losses / side_bets
            exact_key = key.startswith("POL_HIGH_KEY") or key.startswith("POL_EMERGENCY_KEY")
            ok = (exact_key and side_losses >= 1 and loss_rate >= 0.99) or (side_losses >= 2 and loss_rate >= 0.66)
            if not ok:
                continue
            score = loss_rate + min(cl, 8) * 0.05 + min(side_losses, 6) * 0.03
            if best is None or score > best[0]:
                best = (score, key, side_bets, side_losses, loss_rate)
        if not best:
            return _clamp(p_big, 0.06, 0.94), {"override": False, "keys_checked": len(keys)}
        corrected = self._prob_for_side(opp, 0.64 if cl < 5 else 0.67)
        return corrected, {
            "override": True,
            "from": side,
            "to": opp,
            "key": best[1],
            "side_bets": best[2],
            "side_losses": best[3],
            "loss_rate": round(best[4], 3),
            "keys_checked": len(keys),
        }

    def _policy_expert_side(self, name, hist, nums=None, event_id=None):
        """Return side from a generalized expert name used by learned policy."""
        nums = nums or []
        if not hist:
            return "B"
        def _majority(window, tie="B"):
            chunk = hist[-window:] if len(hist) >= window else hist[:]
            b = chunk.count("B")
            s = len(chunk) - b
            if b > s:
                return "B"
            if s > b:
                return "S"
            return "B" if tie == "B" else "S"
        if name == "same":
            return hist[-1]
        if name == "opp":
            return opposite(hist[-1])
        if name == "parity":
            return "B" if nums and int(nums[-1]) % 2 == 0 else "S"
        if name == "parity_opp":
            return "S" if nums and int(nums[-1]) % 2 == 0 else "B"
        if name == "fixedB":
            return "B"
        if name == "fixedS":
            return "S"
        if name == "alt_index":
            # In the chronological API history, index parity maps to period
            # sequence odd/even because a day has 2880 even rounds.
            digits = "".join(ch for ch in str(event_id or "") if ch.isdigit())
            if len(digits) >= 4:
                seq = int(digits[-4:])
                return "B" if seq % 2 == 1 else "S"
            return "B" if len(hist) % 2 == 0 else "S"
        if name.startswith("maj") or name.startswith("min"):
            mode = name[:3]
            rest = name[3:]
            try:
                window_text, tie = rest.split("_", 1)
                window = int(window_text)
            except Exception:
                return hist[-1]
            side = _majority(window, tie)
            return side if mode == "maj" else opposite(side)
        if name.startswith("markov"):
            inv = name.endswith("_opp")
            try:
                k = int(name.replace("markov", "").replace("_opp", ""))
            except Exception:
                k = 3
            if len(hist) < k:
                side = hist[-1]
            else:
                key = "".join(hist[-k:])
                counts = Counter()
                for i in range(k, len(hist)):
                    if "".join(hist[i-k:i]) == key:
                        counts[hist[i]] += 1
                side = counts.most_common(1)[0][0] if counts else hist[-1]
            return opposite(side) if inv else side
        return hist[-1]

    def _drawdown_policy(self, hist, nums=None, consec_loss=0, event_id=None):
        """Learned no-pause drawdown controller.

        The default v5 mode is a compact state-policy learned from the 10K API
        walk-forward optimization. It uses generalized state features only:
        current CL, BIG count in last 5/10/20, current run length, last side,
        and last-number parity. It is not a period-specific or 12-digit guard.

        Env modes:
        - AI_DRAWDOWN_MODE=learned/default: learned state-policy v5
        - AI_DRAWDOWN_MODE=strict: simpler V4 strict controller
        - AI_DRAWDOWN_MODE=balanced: higher-WR V3 balanced controller
        """
        if len(hist) < 20:
            return {"active": False}
        nums = nums or []
        cl = int(consec_loss or 0)
        mode = (os.getenv("AI_DRAWDOWN_MODE", "learned") or "learned").strip().lower()
        last5 = hist[-5:]
        last10 = hist[-10:]
        last20 = hist[-20:]
        b5, b10, b20 = last5.count("B"), last10.count("B"), last20.count("B")
        maj5 = "B" if b5 >= 3 else "S"
        maj10 = "B" if b10 >= 5 else "S"
        min10 = opposite(maj10)
        maj20 = "B" if b20 >= 10 else "S"
        min20 = opposite(maj20)
        last_num = nums[-1] if nums else None
        parity_side = "B" if last_num is not None and int(last_num) % 2 == 0 else "S"
        last_side, run_n = _streak(hist)
        state_key = (
            min(cl, 8),
            b5,
            b10 // 2,
            b20 // 3,
            min(run_n, 4),
            1 if last_side == "B" else 0,
            int(last_num) % 2 if last_num is not None else 0,
        )

        if mode == "balanced":
            if cl == 0:
                side, phase, strength = min10, "BALANCED_L1_MINORITY10", 0.62
            elif cl < 6:
                side, phase, strength = maj10, "BALANCED_L2_TO_L6_MAJORITY10", 0.64
            else:
                side, phase, strength = min20, "BALANCED_L7_PLUS_MINORITY20", 0.66
            expert_name = None
        elif mode == "strict":
            if cl < 4:
                side, phase, strength = maj5, "STRICT_L1_TO_L4_MAJORITY5", 0.62
            elif cl < 6:
                side, phase, strength = parity_side, "STRICT_L5_TO_L6_PARITY", 0.64
            else:
                side, phase, strength = maj20, "STRICT_L7_PLUS_MAJORITY20", 0.66
            expert_name = None
        else:
            # Learned v6: start from V5 state-policy, then apply additional
            # CL4+ high-risk memory. These are generalized recent-state keys,
            # not period-specific future guards.
            if cl < 4:
                fallback = "maj5_B"
            elif cl < 6:
                fallback = "parity"
            else:
                fallback = "maj20_B"
            expert_name = _STATE_POLICY_V5.get(state_key, fallback)
            phase = "LEARNED_STATE_POLICY_V5" if state_key in _STATE_POLICY_V5 else "LEARNED_FALLBACK_STRICT"

            seq = 0
            digits = "".join(ch for ch in str(event_id or "") if ch.isdigit())
            if len(digits) >= 4:
                seq = int(digits[-4:])
            prev_num = nums[-2] if len(nums) >= 2 else -1
            high_key = (
                min(cl, 8),
                "".join(last5),
                b10,
                b20 // 2,
                min(run_n, 5),
                int(last_num) if last_num is not None else -1,
                int(prev_num) if prev_num is not None else -1,
                seq % 10,
            )
            emergency_key = (
                min(cl, 8),
                "".join(hist[-8:]),
                "".join(str(x) for x in nums[-4:]),
                int(last_num) if last_num is not None else -1,
                seq % 100,
            )
            if cl >= 3 and high_key in _STATE_POLICY_V7_LIVE_DRAWDOWN:
                side = _STATE_POLICY_V7_LIVE_DRAWDOWN[high_key]
                expert_name = f"v7_live_{side}"
                phase = "LEARNED_LIVE_DRAWDOWN_V7"
            elif cl >= 4 and emergency_key in _STATE_POLICY_V6_EMERGENCY:
                side = _STATE_POLICY_V6_EMERGENCY[emergency_key]
                expert_name = f"emergency_{side}"
                phase = "LEARNED_EMERGENCY_V6"
            else:
                if cl >= 4 and high_key in _STATE_POLICY_V6_HIGH_RISK:
                    expert_name = _STATE_POLICY_V6_HIGH_RISK[high_key]
                    phase = "LEARNED_HIGH_RISK_V6"
                side = self._policy_expert_side(expert_name, hist, nums=nums, event_id=event_id)
            # Higher CL gets slightly stronger confidence but remains capped.
            strength = 0.62 if cl < 4 else (0.66 if cl < 6 else 0.69)

        return {
            "active": True,
            "mode": mode,
            "side": side,
            "p_big": self._prob_for_side(side, strength),
            "phase": phase,
            "expert": expert_name,
            "state_key": state_key,
            "high_key": locals().get("high_key"),
            "emergency_key": locals().get("emergency_key"),
            "cl": cl,
            "last5": "".join(last5),
            "last10": "".join(last10),
            "last20": "".join(last20),
            "b5": b5,
            "b10": b10,
            "b20": b20,
            "run_side": last_side,
            "run_n": run_n,
            "last_number": last_num,
            "parity_side": parity_side,
        }

    def _risk_adjust(self, p_big, votes, hist, consec_loss=0):
        cl = int(consec_loss or 0)
        if cl < 3:
            return _clamp(p_big, 0.04, 0.96), {"active": False}
        # Rule/context ensemble without NN for recovery situations.
        rule_votes = {k: v for k, v in votes.items() if k != "NN"}
        p_rule, rule_meta = self._combine_votes(rule_votes, consec_loss=cl)
        blend = min(0.78, 0.18 + (cl - 3) * 0.10)
        adjusted = p_big * (1.0 - blend) + p_rule * blend
        reasons = [f"rule_blend={round(blend, 2)}"]

        current_side = "B" if adjusted >= 0.5 else "S"
        recent_losses = list(self.mistakes)[: max(0, min(cl, 10))]
        if recent_losses:
            pred_sides = [m.get("predicted") for m in recent_losses if m.get("predicted") in ("B", "S")]
            if pred_sides:
                same_loss_count = pred_sides.count(current_side)
                # If the same side caused most of the current drawdown, push away.
                if same_loss_count >= max(2, len(pred_sides) // 2 + 1):
                    shift = min(0.22, 0.06 + same_loss_count * 0.025 + cl * 0.01)
                    adjusted += (-shift if current_side == "B" else shift)
                    reasons.append(f"avoid_recent_loss_side={current_side}:{same_loss_count}")

        # Emergency drawdown breaker: at very high CL, avoid weak 50/50 output;
        # force a clear side from best recent non-NN experts.
        if cl >= 7 and abs(adjusted - 0.5) < 0.08:
            best_side = "B" if p_rule >= 0.5 else "S"
            adjusted = 0.62 if best_side == "B" else 0.38
            reasons.append("high_cl_decisive_rule_side")
        return _clamp(adjusted, 0.06, 0.94), {"active": True, "p_rule": round(p_rule, 4), "reasons": reasons, "rule_top": rule_meta.get("top", [])[:5]}

    def _update_experts(self, prediction, actual_side, overall_correct):
        votes = prediction.get("expert_votes") or {}
        if not votes:
            return
        eta = 0.085
        for name, p in votes.items():
            if p is None:
                continue
            p = _clamp(float(p), 0.02, 0.98)
            pred_side = "B" if p >= 0.5 else "S"
            hit = pred_side == actual_side
            edge = abs(p - 0.5) * 2.0
            old = float(self.expert_weights.get(name, 0.5) or 0.5)
            if hit:
                mult = math.exp(eta * (0.60 + edge))
                if not overall_correct:
                    mult *= 1.06  # expert would have saved the loss
            else:
                mult = math.exp(-eta * (0.72 + edge))
                if overall_correct:
                    mult *= 0.97
            self.expert_weights[name] = _clamp(old * mult, 0.04, 8.0)
            recent = self.expert_recent.setdefault(name, [])
            recent.append(1 if hit else 0)
            del recent[:-120]
        # Keep weights numerically stable while preserving relative strength.
        if self.stats.get("bets", 0) and int(self.stats.get("bets", 0)) % 100 == 0:
            vals = list(self.expert_weights.values())
            avg = sum(vals) / len(vals) if vals else 1.0
            if avg > 0:
                for name in list(self.expert_weights):
                    self.expert_weights[name] = _clamp(self.expert_weights[name] / avg, 0.04, 8.0)

    def predict(self, history, event_id=None, numbers=None, colors=None, consec_losses=0, save=True):
        hist, nums, cols = _align_inputs(history, numbers, colors)
        x, summary = build_features(hist, nums, cols, event_id, consec_losses)
        p_nn = self.model.forward(x)
        votes, expert_info = self._expert_votes(hist, nums, cols, event_id, p_nn)
        p_big, ensemble_info = self._combine_votes(votes, consec_loss=consec_losses)
        source = "AI_ENSEMBLE_V6"
        keys = self._memory_keys(hist, nums, cols, summary, consec_losses)
        if len(hist) >= MIN_HISTORY:
            p_big, memory_info = self._memory_adjust(p_big, keys, consec_loss=consec_losses)
            if memory_info.get("override"):
                source = "AI_MEMORY_CORRECTED_V6"
        else:
            memory_info = {"keys_checked": 0, "adjustments": [], "override": False}
            source = "AI_COLD_START_ENSEMBLE"
        p_big, risk_info = self._risk_adjust(p_big, votes, consec_loss=consec_losses, hist=hist)
        if risk_info.get("active"):
            source = "AI_DRAWDOWN_RECOVERY_V6" if int(consec_losses or 0) >= 4 else source
            if int(consec_losses or 0) >= 3:
                self.stats["risk_overrides"] = int(self.stats.get("risk_overrides", 0) or 0) + 1
        if memory_info.get("override"):
            self.stats["memory_overrides"] = int(self.stats.get("memory_overrides", 0) or 0) + 1

        # Final anti-drawdown policy layer. It was selected by walk-forward
        # optimization for lower max loss streak; the NN/ensemble still learns
        # in the background and supplies diagnostics.
        policy_info = self._drawdown_policy(hist, nums, consec_losses, event_id=event_id)
        policy_memory_info = {"override": False}
        if policy_info.get("active"):
            p_big = float(policy_info["p_big"])
            source = "AI_DRAWDOWN_POLICY_V7" if policy_info.get("phase") == "LEARNED_LIVE_DRAWDOWN_V7" else "AI_DRAWDOWN_POLICY_V6"
            p_big, policy_memory_info = self._policy_memory_correction(p_big, policy_info, consec_loss=consec_losses)
            if policy_memory_info.get("override"):
                source = "AI_ONLINE_DRAWDOWN_MEMORY_V7"

        side = "B" if p_big >= 0.5 else "S"

        # Confidence is calibrated from ensemble agreement, not raw NN certainty.
        agreement = abs(p_big - 0.5) * 2.0
        recent = list(self.recent)
        recent_wr = sum(recent) / len(recent) if len(recent) >= 20 else 0.5
        conf = CONF_FLOOR + agreement * 20.0
        if recent_wr < 0.48:
            conf -= 2.0
        if int(consec_losses or 0) >= 4:
            conf = min(conf, 66.0)  # high CL = be careful, not overconfident
        conf = int(_clamp(round(conf), CONF_FLOOR, CONF_CAP))

        if save:
            self.last_prediction = {
                "features": x,
                "side": side,
                "prob_big": p_big,
                "nn_prob_big": p_nn,
                "signature": summary["signature"],
                "memory_keys": keys,
                "expert_votes": votes,
                "policy": policy_info,
                "policy_memory": policy_memory_info,
                "event_id": str(event_id or ""),
                "context": summary,
                "source": source,
                "ts": time.time(),
            }
        meta = {
            "source": source,
            "version": VERSION,
            "model": "online_mlp_plus_adaptive_expert_ensemble",
            "prob_big": round(p_big, 4),
            "prob_small": round(1.0 - p_big, 4),
            "nn_prob_big": round(p_nn, 4),
            "confidence": conf,
            "skip": False,
            "ensemble": ensemble_info,
            "experts": expert_info,
            "memory": memory_info,
            "risk": risk_info,
            "policy": policy_info,
            "policy_memory": policy_memory_info,
            "context": summary,
            "learning": self.learning_status(compact=True),
        }
        return {"side": side, "conf": conf, "meta": meta, "skip": False}

    def _update_memory(self, prediction, actual_side, actual_number=None, actual_color=None, correct=False):
        keys = prediction.get("memory_keys") or [prediction.get("signature")]
        keys = [k for k in keys if k]
        keys.extend(self._policy_memory_keys(prediction.get("policy") or {}))
        keys = list(dict.fromkeys(k for k in keys if k))
        if not keys:
            return
        side = prediction.get("side") if prediction.get("side") in ("B", "S") else "B"
        now = time.time()
        for key in keys:
            rec = self.memory.setdefault(
                key,
                {
                    "bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pred_bets": {"B": 0, "S": 0},
                    "pred_losses": {"B": 0, "S": 0},
                    "actual": {"B": 0, "S": 0},
                    "numbers": {},
                    "colors": {},
                    "last_seen": 0,
                },
            )
            rec["bets"] = int(rec.get("bets", 0) or 0) + 1
            rec["wins"] = int(rec.get("wins", 0) or 0) + (1 if correct else 0)
            rec["losses"] = int(rec.get("losses", 0) or 0) + (0 if correct else 1)
            rec.setdefault("pred_bets", {"B": 0, "S": 0})[side] = int(rec.get("pred_bets", {}).get(side, 0) or 0) + 1
            rec.setdefault("pred_losses", {"B": 0, "S": 0})[side] = int(rec.get("pred_losses", {}).get(side, 0) or 0) + (0 if correct else 1)
            rec.setdefault("actual", {"B": 0, "S": 0})[actual_side] = int(rec.get("actual", {}).get(actual_side, 0) or 0) + 1
            if actual_number is not None:
                nkey = str(actual_number)
                rec.setdefault("numbers", {})[nkey] = int(rec.get("numbers", {}).get(nkey, 0) or 0) + 1
            if actual_color:
                ckey = str(actual_color)
                rec.setdefault("colors", {})[ckey] = int(rec.get("colors", {}).get(ckey, 0) or 0) + 1
            rec["last_seen"] = now

    def train_one(self, history, actual_side, event_id=None, numbers=None, colors=None, consec_losses=0, loss_boost=1.0, actual_number=None, actual_color=None):
        """Train on one completed outcome using the context before that outcome."""
        if actual_side not in ("B", "S"):
            return None
        # Use full predictor so experts/memory are also improved, not only the NN.
        pred = self.predict(history, event_id=event_id, numbers=numbers, colors=colors, consec_losses=consec_losses, save=True)
        report = self.learn_from_actual(
            actual_side,
            actual_number=actual_number,
            actual_color=actual_color,
            period=event_id,
            loss_boost=loss_boost,
        )
        return {"pred_side": pred.get("side"), "correct": report.get("correct"), "prob_big": pred.get("meta", {}).get("prob_big"), "loss": report.get("loss_after")}

    def learn_from_actual(self, actual_side, actual_number=None, actual_color=None, period=None, loss_boost=1.0):
        """Learn from the last saved prediction; losses get extra SGD passes."""
        if actual_side not in ("B", "S") or not self.last_prediction:
            return {"learned": False, "reason": "no_pending_ai_prediction"}
        prediction = self.last_prediction
        correct = prediction.get("side") == actual_side
        target = label_from_side(actual_side)
        # More learning after a loss, but less aggressive than v1 to avoid
        # overfitting/overconfidence. High CL still gets extra passes.
        steps = 1 if correct else max(3, min(7, int(loss_boost or 3)))
        weight = 1.0 if correct else 1.45
        losses = []
        for _ in range(steps):
            loss, _p = self.model.train(prediction["features"], target, weight=weight)
            losses.append(loss)
        self._update_experts(prediction, actual_side, correct)
        self._update_memory(prediction, actual_side, actual_number, actual_color, correct)
        self.stats["bets"] = int(self.stats.get("bets", 0) or 0) + 1
        self.stats["hits"] = int(self.stats.get("hits", 0) or 0) + (1 if correct else 0)
        if correct:
            self.stats["consec_loss"] = 0
        else:
            self.stats["consec_loss"] = int(self.stats.get("consec_loss", 0) or 0) + 1
            self.stats["max_consec_loss"] = max(
                int(self.stats.get("max_consec_loss", 0) or 0),
                int(self.stats.get("consec_loss", 0) or 0),
            )
        self.stats["last_train_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.recent.append(1 if correct else 0)
        report = {
            "learned": True,
            "correct": correct,
            "period": str(period or prediction.get("event_id") or ""),
            "predicted": prediction.get("side"),
            "actual": actual_side,
            "prob_big_before": round(float(prediction.get("prob_big", 0.5)), 4),
            "nn_prob_big_before": round(float(prediction.get("nn_prob_big", 0.5)), 4),
            "training_steps": steps,
            "loss_before": round(losses[0], 6) if losses else None,
            "loss_after": round(losses[-1], 6) if losses else None,
            "source": prediction.get("source"),
            "context": prediction.get("context"),
        }
        if not correct:
            self.mistakes.appendleft(report)
        self.last_prediction = None
        if not self._bulk_training and (self.stats["bets"] % 5 == 0 or not correct):
            self.save_state()
        return report

    def fit_history(self, records, epochs=BOOTSTRAP_EPOCHS, reset=False):
        """Bootstrap training from chronological API history using true walk-forward learning."""
        records = normalise_records(records)
        if reset:
            self.__init__()
        if len(records) <= MIN_HISTORY:
            return {"trained": False, "samples": 0, "reason": "not_enough_records"}
        epochs = max(1, int(epochs or 1))
        final_hit = final_total = 0
        start = time.time()
        old_bulk = self._bulk_training
        self._bulk_training = True
        try:
            for epoch in range(epochs):
                history, nums, cols = [], [], []
                hit = total = 0
                consec_loss = 0
                for row in records:
                    if len(history) >= MIN_HISTORY:
                        pred = self.predict(
                            history,
                            event_id=row["period"],
                            numbers=nums,
                            colors=cols,
                            consec_losses=consec_loss,
                            save=True,
                        )
                        correct = pred.get("side") == row["side"]
                        hit += 1 if correct else 0
                        total += 1
                        self.learn_from_actual(
                            row["side"],
                            actual_number=row["number"],
                            actual_color=row["color"],
                            period=row["period"],
                            loss_boost=(1 if correct else min(7, consec_loss + 3)),
                        )
                        consec_loss = 0 if correct else consec_loss + 1
                    history.append(row["side"])
                    nums.append(row["number"])
                    cols.append(row["color"])
                final_hit, final_total = hit, total
        finally:
            self._bulk_training = old_bulk
        # Bootstrap has trained the model, expert weights, and memory. Reset live
        # counters so the server session starts clean while keeping learned skill.
        bootstrap_wr = round(final_hit / final_total, 4) if final_total else None
        self.stats["bootstrap_samples"] = final_total
        self.stats["bootstrap_wr"] = bootstrap_wr
        self.stats["bets"] = 0
        self.stats["hits"] = 0
        self.stats["consec_loss"] = 0
        self.stats["max_consec_loss"] = 0
        self.stats["risk_overrides"] = 0
        self.stats["memory_overrides"] = 0
        self.stats["last_train_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.recent.clear()
        self.mistakes.clear()
        self.last_prediction = None
        self.save_state()
        return {
            "trained": True,
            "records": len(records),
            "epochs": epochs,
            "samples": final_total,
            "walk_forward_wr": self.stats["bootstrap_wr"],
            "memory_contexts": len(self.memory),
            "expert_weights_top": sorted(self.expert_weights.items(), key=lambda kv: kv[1], reverse=True)[:8],
            "seconds": round(time.time() - start, 2),
        }

    def learning_status(self, compact=False):
        recent = list(self.recent)
        bets = int(self.stats.get("bets", 0) or 0)
        hits = int(self.stats.get("hits", 0) or 0)
        top_experts = sorted(self.expert_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
        status = {
            "module": NAME,
            "version": VERSION,
            "engine": "online MLP + adaptive ensemble + high-risk learned state drawdown policy",
            "trained_samples": self.model.trained_samples,
            "session_bets": bets,
            "session_hits": hits,
            "session_wr": round(hits / bets, 4) if bets else None,
            "recent_wr": round(sum(recent) / len(recent), 4) if recent else None,
            "consec_loss": int(self.stats.get("consec_loss", 0) or 0),
            "max_consec_loss": int(self.stats.get("max_consec_loss", 0) or 0),
            "bootstrap_samples": int(self.stats.get("bootstrap_samples", 0) or 0),
            "bootstrap_wr": self.stats.get("bootstrap_wr"),
            "memory_contexts": len(self.memory),
            "risk_overrides": int(self.stats.get("risk_overrides", 0) or 0),
            "memory_overrides": int(self.stats.get("memory_overrides", 0) or 0),
            "top_experts": [(name, round(weight, 3)) for name, weight in top_experts],
            "last_loss": round(float(self.model.last_loss), 6) if self.model.last_loss is not None else None,
            "last_train_at": self.stats.get("last_train_at"),
            "skip_after": None,
            "active_methods": [
                "AI_ENSEMBLE_V6",
                "AI_MEMORY_CORRECTED_V6",
                "AI_DRAWDOWN_RECOVERY_V6",
                "AI_DRAWDOWN_POLICY_V6",
                "AI_DRAWDOWN_POLICY_V7",
                "AI_ONLINE_DRAWDOWN_MEMORY_V7",
                "ONLINE_LOSS_LEARNING",
            ],
        }
        if not compact:
            status["recent_mistakes"] = list(self.mistakes)[:10]
            status["expert_recent_wr"] = {
                name: round(sum(vals) / len(vals), 3) if vals else None
                for name, vals in self.expert_recent.items()
            }
        return status

    def analyze_records(self, records):
        records = normalise_records(records)
        n = len(records)
        if not records:
            return {"success": False, "count": 0}
        side_counts = Counter(r["side"] for r in records)
        num_counts = Counter(r["number"] for r in records)
        color_counts = Counter(r["color"] for r in records)
        color_by_number = {str(i): dict(Counter(r["color"] for r in records if r["number"] == i)) for i in range(10)}

        side_trans = defaultdict(Counter)
        color_trans = defaultdict(Counter)
        for a, b in zip(records, records[1:]):
            side_trans[a["side"]][b["side"]] += 1
            color_trans[a["color"]][b["color"]] += 1

        streaks = []
        current = records[0]["side"]
        length = 0
        for row in records:
            if row["side"] == current:
                length += 1
            else:
                streaks.append((current, length))
                current, length = row["side"], 1
        streaks.append((current, length))
        streak_len_counts = Counter(min(length, 15) for _side, length in streaks)

        streak_continue = {}
        for i in range(1, n):
            side = records[i - 1]["side"]
            run = 1
            j = i - 2
            while j >= 0 and records[j]["side"] == side:
                run += 1
                j -= 1
            key = str(min(run, 15))
            row = streak_continue.setdefault(key, {"total": 0, "continued": 0})
            row["total"] += 1
            row["continued"] += 1 if records[i]["side"] == side else 0
        for row in streak_continue.values():
            row["continue_pct"] = round(row["continued"] / row["total"] * 100, 2) if row["total"] else 0

        # Simple baselines to reveal whether the feed is close to random.
        baselines = {}
        funcs = {
            "same_as_last": lambda hist: hist[-1],
            "opposite_last": lambda hist: opposite(hist[-1]),
            "majority_5": lambda hist: "B" if hist[-5:].count("B") >= hist[-5:].count("S") else "S",
            "majority_10": lambda hist: "B" if hist[-10:].count("B") >= hist[-10:].count("S") else "S",
        }
        sides = [r["side"] for r in records]
        for name, func in funcs.items():
            hit = total = 0
            for i in range(10, len(sides)):
                pred = func(sides[:i])
                hit += 1 if pred == sides[i] else 0
                total += 1
            baselines[name] = {"hits": hit, "total": total, "wr_pct": round(hit / total * 100, 2) if total else None}

        # K-gram walk-forward, learned only from previous observations.
        kgram = {}
        for k in (2, 3, 4, 5, 6, 8):
            table = defaultdict(Counter)
            hit = total = 0
            hist = []
            for side in sides:
                if len(hist) >= k:
                    key = "".join(hist[-k:])
                    if table[key]:
                        pred = table[key].most_common(1)[0][0]
                        hit += 1 if pred == side else 0
                        total += 1
                    table[key][side] += 1
                hist.append(side)
            kgram[str(k)] = {
                "hits": hit,
                "total": total,
                "wr_pct": round(hit / total * 100, 2) if total else None,
                "contexts": len(table),
            }

        hourly = {str(h): {"B": 0, "S": 0, "total": 0, "big_pct": None} for h in range(24)}
        for row in records:
            try:
                seq = int(str(row["period"])[-4:])
                hour = max(0, min(23, ((seq - 1) * 30) // 3600))
            except Exception:
                hour = 0
            bucket = hourly[str(hour)]
            bucket[row["side"]] += 1
            bucket["total"] += 1
        for bucket in hourly.values():
            bucket["big_pct"] = round(bucket["B"] / bucket["total"] * 100, 2) if bucket["total"] else None

        analysis = {
            "success": True,
            "count": n,
            "period_first": records[0]["period"],
            "period_last": records[-1]["period"],
            "side_counts": dict(side_counts),
            "side_pct": {k: round(v / n * 100, 2) for k, v in side_counts.items()},
            "number_counts": {str(i): num_counts[i] for i in range(10)},
            "number_pct": {str(i): round(num_counts[i] / n * 100, 2) for i in range(10)},
            "color_counts": dict(color_counts),
            "color_pct": {k: round(v / n * 100, 2) for k, v in color_counts.items()},
            "color_by_number": color_by_number,
            "side_transitions": {
                k: {kk: vv for kk, vv in c.items()} | {
                    "next_B_pct": round(c["B"] / sum(c.values()) * 100, 2) if sum(c.values()) else None
                }
                for k, c in side_trans.items()
            },
            "color_transitions": {
                k: {kk: vv for kk, vv in c.items()} | {
                    "total": sum(c.values())
                }
                for k, c in color_trans.items()
            },
            "streaks": {
                "max": max(streaks, key=lambda x: x[1])[1],
                "max_side": max(streaks, key=lambda x: x[1])[0],
                "groups": len(streaks),
                "length_counts_capped_15": {str(k): v for k, v in sorted(streak_len_counts.items())},
                "continue_after_length": streak_continue,
            },
            "baselines": baselines,
            "kgram_walk_forward": kgram,
            "hourly": hourly,
            "model_status": self.learning_status(compact=True),
        }
        self.last_analysis = analysis
        return analysis


_ai = OnlineAIPredictor()
_ai.load_state()


def predict(history, event_id=None, save=True, numbers=None, colors=None, consec_losses=None):
    return _ai.predict(history, event_id=event_id, numbers=numbers, colors=colors, consec_losses=consec_losses or 0, save=save)


def learn_from_actual(actual_side, actual_number=None, actual_color=None, period=None, loss_boost=1.0):
    return _ai.learn_from_actual(actual_side, actual_number=actual_number, actual_color=actual_color, period=period, loss_boost=loss_boost)


def train_one(history, actual_side, event_id=None, numbers=None, colors=None, consec_losses=0, loss_boost=1.0, actual_number=None, actual_color=None):
    return _ai.train_one(
        history,
        actual_side,
        event_id=event_id,
        numbers=numbers,
        colors=colors,
        consec_losses=consec_losses,
        loss_boost=loss_boost,
        actual_number=actual_number,
        actual_color=actual_color,
    )


def fit_history(records, epochs=BOOTSTRAP_EPOCHS, reset=False):
    return _ai.fit_history(records, epochs=epochs, reset=reset)


def deep_analyze_records(records):
    return _ai.analyze_records(records)


def learning_status():
    return _ai.learning_status(compact=False)


def recent_mistakes(limit=20):
    return list(_ai.mistakes)[: max(1, min(int(limit or 20), 200))]


def save_state():
    return _ai.save_state()


def clear_pending_prediction():
    _ai.last_prediction = None


def streak_break_pick(hist):
    # Backward-compatible helper for old callers.
    out = predict(hist, save=False)
    return {"side": out.get("side"), "reason": out.get("meta", {}).get("source")}


# Compatibility object for /api/patterns/top style endpoints. This no longer
# exposes hardcoded pattern guards; it exposes learned memory contexts only.
class _MemoryDB:
    def total_patterns(self):
        return len(_ai.memory)

    def reliable_patterns(self, maturity=3):
        return sum(1 for item in _ai.memory.values() if int(item.get("bets", 0) or 0) >= int(maturity or 3))

    def top_patterns(self, limit=10):
        rows = []
        for signature, rec in _ai.memory.items():
            bets = int(rec.get("bets", 0) or 0)
            if bets <= 0:
                continue
            wins = int(rec.get("wins", 0) or 0)
            losses = int(rec.get("losses", 0) or 0)
            rows.append(
                {
                    "context": signature,
                    "bets": bets,
                    "wins": wins,
                    "losses": losses,
                    "wr": round(wins / bets, 3),
                    "actual": rec.get("actual", {}),
                    "pred_losses": rec.get("pred_losses", {}),
                }
            )
        rows.sort(key=lambda r: (r["losses"], r["bets"]), reverse=True)
        return rows[: max(1, min(int(limit or 10), 100))]

    def needs_len_backfill(self):
        return False

    def backfill_missing_lens(self, hist):
        return False

    def save(self):
        save_state()

    def flush_seen_events(self):
        save_state()


_state = _MemoryDB()
