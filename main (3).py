# main.py — DRAGO Orihost v36 (Recovery4/Deep3/L7Guard, no SKIP)
# pattern.py = v36-recovery4-deep3-l7guard-noskip
# Level system remains active. Every valid round publishes BIG/SMALL.
# Auth: X-VPS-Key must match VPS_SECRET in .env.

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load_dotenv_file():
    """Load a local .env without overriding host-provided environment values."""
    for env_path in (Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"):
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Orihost par local .env authoritative hai (original server behaviour).
            # Isse stale OS-level PORT/VPS_SECRET values connection nahi todte.
            if key:
                os.environ[key] = value
        return str(env_path)
    return None


_ENV_FILE = load_dotenv_file()

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import pattern

CALCS = [pattern]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drago")

FIREBASE_AUTOBET_URL = os.getenv(
    "FIREBASE_AUTOBET_URL", "https://auto-bet-pro-default-rtdb.firebaseio.com"
)
FIREBASE_AUTOBET_SECRET = os.getenv("FIREBASE_AUTOBET_SECRET", "")
FIREBASE_URL = os.getenv(
    "FIREBASE_URL", "https://drago-predictor-default-rtdb.firebaseio.com"
)
FIREBASE_SECRET = os.getenv("FIREBASE_SECRET", "")

STREAM_ID = "wingo_30s"
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
POLL_SEC = int(os.getenv("POLL_SEC", "3") or 3)
HISTORY_MAX = int(os.getenv("HISTORY_MAX", "5000") or 5000)
SCORE_WINDOW = 40
MIN_PAPER_BETS = 15
CONF_FLOOR = 56
CONF_CAP = 66
REGIME_WINDOW = 20
PACK_GUARD_N = 6
MAX_LEVEL = 6
HIGH_LEVEL_FROM = 4

# Level system stays enabled. LEVEL_CAP=0 means no display cap (actual level).
LEVEL_CAP = int(os.getenv("LEVEL_CAP", "0") or 0)
STOP_LOSS_STREAK = int(os.getenv("STOP_LOSS_STREAK", "9") or 9)
STREAK_BREAK_AFTER = 99999
STREAK_BREAK_CONF = 66

MISTAKES_FILE = os.getenv("MISTAKES_FILE", "mistakes.jsonl")
MISTAKES_MAX_LINES = int(os.getenv("MISTAKES_MAX_LINES", "500") or 500)
LEVEL_STATE_FILE = os.getenv("LEVEL_STATE_FILE", "level_state.json")
DRAW_STORE_MAX = int(os.getenv("DRAW_STORE_MAX", "100000") or 100000)
DRAW_STORE_FILE = os.getenv("DRAW_STORE_FILE", "wingo30s_history.json")

# Current Orihost prediction-server port. .env PORT/SERVER_PORT can override it.
PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "30252")))
VPS_SECRET = (os.getenv("VPS_SECRET") or os.getenv("DRAGO_VPS_SECRET") or "").strip()
LOGS_SECRET = (os.getenv("LOGS_SECRET") or os.getenv("MISTAKES_LOG_KEY") or "").strip()
RENDER_KEEPALIVE_URL = (
    os.getenv("RENDER_KEEPALIVE_URL") or "https://dragopredictor.onrender.com"
).rstrip("/")
RENDER_KEEPALIVE_SEC = int(os.getenv("RENDER_KEEPALIVE_SEC", "180") or 180)

# Original fallback values retained for deployment compatibility; .env takes priority.
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or "8782822978:AAEitqI-CdxbiAN3-55Ltf72i79BpIjBeaA"
).strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "6656009938").strip()
TELEGRAM_LEVEL_ALERT_AT = int(os.getenv("TELEGRAM_LEVEL_ALERT_AT", "5") or 5)
DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR", "21") or 21)
SESSION_END_HOUR = int(os.getenv("SESSION_END_HOUR", "21") or 21)
HOURLY_MIN_SAMPLE = int(os.getenv("HOURLY_MIN_SAMPLE", "10") or 10)

_level_lock = threading.Lock()
_mistakes_lock = threading.Lock()
_draw_store_lock = threading.Lock()
_hourly_lock = threading.Lock()
_draw_store = []
_hourly_stats = {}
_tg_last_level_alert = -1


def ist_now():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def ist_today_str():
    return ist_now().strftime("%Y-%m-%d")


def risk_status(consec):
    if consec >= 8:
        return "🔴 CRITICAL"
    if consec >= 6:
        return "🟠 HIGH"
    if consec >= 4:
        return "🟡 MEDIUM"
    return "🟢 NORMAL"


def telegram_send(text, silent=False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("telegram not configured; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": str(text)[:3900],
                "disable_notification": bool(silent),
            },
            timeout=15,
        )
        if response.status_code != 200:
            log.warning("telegram HTTP %s %s", response.status_code, response.text[:200])
            return False
        return True
    except Exception as exc:
        log.warning("telegram send failed: %s", exc)
        return False


def reset_tg_level_alert():
    global _tg_last_level_alert
    _tg_last_level_alert = -1


def telegram_level_alert(consec, level, pred, actual, period, method):
    """Alert when the displayed level reaches configured threshold."""
    global _tg_last_level_alert
    # Ek loss streak mein sirf ek high-level alert. CL8 ke baad CL9/CL10
    # hone par duplicate alert mat bhejo; win ke baad reset_tg_level_alert()
    # next streak ke liye alert dobara enable karega.
    if level < TELEGRAM_LEVEL_ALERT_AT or _tg_last_level_alert >= 0:
        return
    _tg_last_level_alert = consec
    telegram_send(
        "\n".join(
            [
                "🚨 DRAGO HIGH-LEVEL ALERT",
                "━━━━━━━━━━━━━━━━━━",
                f"📈 Current Level: L{level}",
                f"❌ Consecutive Losses: {consec}",
                f"⚠️ Risk: {risk_status(consec)}",
                "",
                f"🎯 Prediction: {pred}",
                f"📌 Actual: {actual}",
                f"🧠 Method: {method}",
                f"🔢 Period: {period}",
                f"🕐 Time: {ist_now().strftime('%d-%m-%Y %H:%M:%S')} IST",
                "━━━━━━━━━━━━━━━━━━",
                "Next prediction will use deep-recovery calculation.",
            ]
        )
    )


def telegram_recovery_alert(consec_before, level_before, period, method):
    if level_before < TELEGRAM_LEVEL_ALERT_AT:
        return
    telegram_send(
        "\n".join(
            [
                "✅ DRAGO RECOVERY WIN",
                "━━━━━━━━━━━━━━━━━━",
                f"Recovered at: L{level_before}",
                f"Previous consecutive losses: {consec_before}",
                f"Period: {period}",
                f"Method: {method}",
                "Current level reset: L1",
                f"Time: {ist_now().strftime('%d-%m-%Y %H:%M:%S')} IST",
            ]
        )
    )


def record_hourly_result(correct, consec_after):
    now = ist_now()
    day, hour = now.strftime("%Y-%m-%d"), now.hour
    with _hourly_lock:
        day_stats = _hourly_stats.setdefault(
            day, {h: {"wins": 0, "losses": 0, "max_consec": 0} for h in range(24)}
        )
        bucket = day_stats[hour]
        bucket["wins" if correct else "losses"] += 1
        bucket["max_consec"] = max(bucket["max_consec"], int(consec_after))


def build_daily_report(day=None):
    """Clear report using completed hours only; partial current hour is excluded."""
    day = day or ist_today_str()
    with _hourly_lock:
        hours = dict(_hourly_stats.get(day) or {})
    if not hours:
        return f"📊 DRAGO DAILY REPORT\n📅 {day} IST\nNo settled predictions."

    now = ist_now()
    configured_end = max(0, min(24, SESSION_END_HOUR))
    # At 21:xx include through 20:59. Never compare a partial hour with full hours.
    completed_end = min(configured_end, now.hour)
    rows = []
    for hour in range(completed_end):
        bucket = hours.get(hour) or {"wins": 0, "losses": 0, "max_consec": 0}
        wins, losses = int(bucket["wins"]), int(bucket["losses"])
        total = wins + losses
        if total:
            rows.append((hour, wins, losses, total, wins / total, int(bucket["max_consec"])))

    if not rows:
        return f"📊 DRAGO DAILY REPORT\n📅 {day} IST\nNo completed-hour data yet."

    total_wins = sum(row[1] for row in rows)
    total_losses = sum(row[2] for row in rows)
    total = total_wins + total_losses
    overall_wr = total_wins / total if total else 0
    eligible = [row for row in rows if row[3] >= HOURLY_MIN_SAMPLE]
    best = max(eligible, key=lambda row: (row[4], row[3])) if eligible else None
    worst = min(eligible, key=lambda row: (row[4], -row[3])) if eligible else None
    first_hour, last_hour = rows[0][0], rows[-1][0]
    max_cl = ENGINE.loss.max_consec_today

    lines = [
        "📊 DRAGO DAILY REPORT",
        "━━━━━━━━━━━━━━━━━━",
        f"📅 Date: {day} IST",
        f"⏰ Completed data: {first_hour:02d}:00–{last_hour:02d}:59",
        "",
        f"🎮 Total Bets: {total}",
        f"✅ Wins: {total_wins}",
        f"❌ Losses: {total_losses}",
        f"🎯 Win Rate: {overall_wr * 100:.1f}%",
        "",
        f"🔥 Maximum Consecutive Loss: {max_cl}",
        f"⚠️ Risk Status: {risk_status(max_cl)}",
        f"📈 Highest Displayed Level: {ENGINE.loss.max_badge()}",
    ]
    if best:
        lines += [
            "",
            "🟢 BEST COMPLETED HOUR",
            f"{best[0]:02d}:00–{best[0]:02d}:59 | Bets {best[3]} | ✅ {best[1]} | ❌ {best[2]}",
            f"WR {best[4] * 100:.1f}% | Max CL {best[5]}",
        ]
    if worst:
        lines += [
            "",
            "🔴 WEAKEST COMPLETED HOUR",
            f"{worst[0]:02d}:00–{worst[0]:02d}:59 | Bets {worst[3]} | ✅ {worst[1]} | ❌ {worst[2]}",
            f"WR {worst[4] * 100:.1f}% | Max CL {worst[5]}",
        ]
    lines += ["", "🕐 HOUR-BY-HOUR"]
    for hour, wins, losses, total_h, wr, max_hour_cl in rows:
        lines.append(
            f"{hour:02d}h │ Bets {total_h} │ ✅ {wins} │ ❌ {losses} │ "
            f"WR {wr * 100:.1f}% │ CL {max_hour_cl}"
        )
    lines += ["", f"ℹ️ {now.hour:02d}h partial hour totals mein include nahi hai."]
    return "\n".join(lines)


def daily_report_loop():
    last_sent = None
    while True:
        try:
            now = ist_now()
            key = f"{now.strftime('%Y-%m-%d')}-{DAILY_REPORT_HOUR}"
            if now.hour == DAILY_REPORT_HOUR and now.minute < 5 and last_sent != key:
                ok = telegram_send(build_daily_report(now.strftime("%Y-%m-%d")))
                log.info("daily report sent=%s key=%s", ok, key)
                if ok:
                    last_sent = key
            time.sleep(30)
        except Exception as exc:
            log.error("daily report loop: %s", exc)
            time.sleep(60)


def number_to_size_letter(number):
    return "B" if int(number) >= 5 else "S"


def letter_to_label(letter):
    return {"B": "BIG", "S": "SMALL"}.get(letter, "WAIT")


def period_sort_key(period):
    text = str(period).strip()
    digits = "".join(char for char in text if char.isdigit())
    if digits:
        try:
            return 0, int(digits)
        except Exception:
            pass
    return 1, text


def next_period(period):
    text = str(period).strip()
    if not text.isdigit() or len(text) != 17:
        return str(int(text) + 1) if text.isdigit() else text
    date_part, middle, sequence = text[:8], text[8:13], int(text[13:])
    if sequence < 2880:
        return f"{date_part}{middle}{sequence + 1:04d}"
    next_day = datetime.strptime(date_part, "%Y%m%d") + timedelta(days=1)
    return f"{next_day.strftime('%Y%m%d')}{middle}0001"


def fb_put(base, path, data, secret=""):
    if not base:
        return
    try:
        url = f"{base.rstrip('/')}/{path}.json" + (f"?auth={secret}" if secret else "")
        response = requests.put(url, json=data, timeout=15)
        if response.status_code not in (200, 201):
            log.warning("firebase PUT %s -> %s %s", path, response.status_code, response.text[:160])
    except Exception as exc:
        log.error("firebase PUT error: %s", exc)


def fb_delete(base, path, secret=""):
    if not base:
        return
    try:
        url = f"{base.rstrip('/')}/{path}.json" + (f"?auth={secret}" if secret else "")
        requests.delete(url, timeout=20)
    except Exception as exc:
        log.error("firebase DELETE error: %s", exc)


def append_mistake(record):
    slim = {
        "period": record.get("period"),
        "prediction": record.get("prediction"),
        "actual": record.get("actual"),
        "method": record.get("method") or "UNKNOWN",
    }
    try:
        with _mistakes_lock:
            with open(MISTAKES_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
            if MISTAKES_MAX_LINES > 0:
                with open(MISTAKES_FILE, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
                if len(lines) > MISTAKES_MAX_LINES:
                    with open(MISTAKES_FILE, "w", encoding="utf-8") as handle:
                        handle.writelines(lines[-MISTAKES_MAX_LINES:])
    except Exception as exc:
        log.error("mistake log error: %s", exc)


class LossLevel:
    def __init__(self):
        self.consec_losses = 0
        self.wins = 0
        self.losses = 0
        self.max_consec_today = 0
        self.daily_date = ist_today_str()

    def _rollover_day(self):
        today = ist_today_str()
        if self.daily_date != today:
            self.daily_date = today
            self.max_consec_today = self.consec_losses

    def to_dict(self):
        return {
            "consec_losses": self.consec_losses,
            "wins": self.wins,
            "losses": self.losses,
            "max_consec_today": self.max_consec_today,
            "daily_date": self.daily_date,
            "ts_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save_state(self):
        try:
            with _level_lock:
                temporary = LEVEL_STATE_FILE + ".tmp"
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(self.to_dict(), handle, ensure_ascii=False)
                os.replace(temporary, LEVEL_STATE_FILE)
        except Exception as exc:
            log.warning("level state save: %s", exc)

    def load_state(self):
        try:
            if not os.path.exists(LEVEL_STATE_FILE):
                return
            with _level_lock:
                with open(LEVEL_STATE_FILE, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            self.consec_losses = int(data.get("consec_losses", 0) or 0)
            self.wins = int(data.get("wins", 0) or 0)
            self.losses = int(data.get("losses", 0) or 0)
            self.max_consec_today = int(data.get("max_consec_today", 0) or 0)
            self.daily_date = str(data.get("daily_date") or ist_today_str())
            self._rollover_day()
            log.info("level state loaded: consec=%s level=%s", self.consec_losses, self.level)
        except Exception as exc:
            log.warning("level state load: %s", exc)

    def on_result(self, correct):
        self._rollover_day()
        if correct:
            self.wins += 1
            self.consec_losses = 0
        else:
            self.losses += 1
            self.consec_losses += 1
            self.max_consec_today = max(self.max_consec_today, self.consec_losses)
        self.save_state()

    def touch(self):
        self._rollover_day()
        self.max_consec_today = max(self.max_consec_today, self.consec_losses)

    @property
    def level(self):
        raw = self.consec_losses + 1
        return min(raw, LEVEL_CAP) if LEVEL_CAP > 0 else raw

    @property
    def max_level_today(self):
        raw = self.max_consec_today + 1
        return min(raw, LEVEL_CAP) if LEVEL_CAP > 0 else raw

    @property
    def stop_loss_hit(self):
        return STOP_LOSS_STREAK > 0 and self.consec_losses >= STOP_LOSS_STREAK

    def badge(self):
        return f"L{self.level}"

    def max_badge(self):
        return f"L{self.max_level_today}"

    def is_at_or_beyond_old_cap(self):
        return self.consec_losses >= MAX_LEVEL


class Engine:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_MAX)
        self.history_nums = deque(maxlen=HISTORY_MAX)
        self.last_period = None
        self.bootstrapped = False
        self.paper = defaultdict(lambda: deque(maxlen=SCORE_WINDOW))
        self.pending_calc = {}
        self.pending_final = None
        self.loss = LossLevel()
        self.total_final = 0
        self.wins_final = 0
        self.draw_count = 0
        self.latest = {
            "stream": STREAM_ID,
            "period": "",
            "prediction": "WAIT",
            "confidence": 0,
            "level": 1,
            "badge": "L1",
            "consec_losses": 0,
            "method": "INIT",
            "status": "WAIT",
            "updated_at": "",
        }

    def calc_weight(self, name):
        records = list(self.paper[name])
        if len(records) < MIN_PAPER_BETS:
            return 1.0
        win_rate = sum(records) / len(records)
        weight = max(0.05, win_rate**2)
        miss_streak = 0
        for result in reversed(records):
            if result:
                break
            miss_streak += 1
        if miss_streak >= 3:
            weight *= 0.35
        elif miss_streak == 2:
            weight *= 0.60
        return weight

    def all_weights(self):
        return {module.NAME: round(self.calc_weight(module.NAME), 4) for module in CALCS}

    def _regime_b_ratio(self):
        recent = list(self.history)[-REGIME_WINDOW:]
        return recent.count("B") / len(recent) if len(recent) >= 12 else None

    def settle(self, actual):
        for name, side in list(self.pending_calc.items()):
            if side in ("B", "S"):
                self.paper[name].append(side == actual)
        self.pending_calc = {}
        if not self.pending_final or self.pending_final.get("side") not in ("B", "S"):
            return

        pending = self.pending_final
        predicted = pending["side"]
        correct = predicted == actual
        self.total_final += 1
        self.wins_final += int(correct)
        level_before = self.loss.level
        consec_before = self.loss.consec_losses
        self.loss.on_result(correct)
        record_hourly_result(correct, self.loss.consec_losses)

        if correct:
            telegram_recovery_alert(
                consec_before, level_before, str(pending.get("period") or ""),
                str(pending.get("source") or pending.get("method") or "UNKNOWN"),
            )
            reset_tg_level_alert()
        else:
            method = str(pending.get("source") or pending.get("method") or "UNKNOWN")
            telegram_level_alert(
                self.loss.consec_losses, self.loss.level, predicted, actual,
                str(pending.get("period") or ""), method,
            )
            append_mistake(
                {
                    "period": pending.get("period"),
                    "prediction": predicted,
                    "actual": actual,
                    "method": method,
                }
            )

        log.info(
            "FINAL settle %s -> %s | pred=%s actual=%s | level=%s consec=%s",
            pending.get("period"), "WIN" if correct else "LOSS", predicted, actual,
            self.loss.badge(), self.loss.consec_losses,
        )
        self.pending_final = None

    def run_calcs(self):
        history = list(self.history)
        numbers = list(self.history_nums)
        votes = {}
        for module in CALCS:
            try:
                # External level synchronization keeps Recovery8 correct after restarts.
                output = module.predict(
                    history,
                    event_id=self.last_period,
                    numbers=numbers,
                    consec_losses=self.loss.consec_losses,
                )
            except Exception as exc:
                log.error("calc %s error: %s", module.NAME, exc)
                output = None
            if output and output.get("side") in ("B", "S"):
                votes[module.NAME] = {
                    "side": output["side"],
                    "conf": int(output.get("conf", CONF_FLOOR)),
                    "meta": output.get("meta") or {},
                }
                self.pending_calc[module.NAME] = output["side"]
            else:
                votes[module.NAME] = None
                self.pending_calc[module.NAME] = None
        return votes

    def decide(self, votes):
        weights = self.all_weights()
        vote = votes.get(pattern.NAME)
        if vote and vote.get("side") in ("B", "S"):
            meta = vote.get("meta") or {}
            source = meta.get("source") or "PATTERN"
            confidence = max(CONF_FLOOR, min(int(vote.get("conf", CONF_FLOOR)), CONF_CAP))
            return {
                "side": vote["side"],
                "conf": confidence,
                "method": source,
                "source": source,
                "skip": False,
                "weights": weights,
                "detail": (
                    f"{self.loss.badge()} conf={confidence} src={source} "
                    f"base={meta.get('base_pattern', 'NO_SKIP')}"
                ),
            }

        # No-SKIP emergency fallback: calculation failure still returns a side.
        history = list(self.history)
        side = history[-1] if history else "B"
        return {
            "side": side,
            "conf": CONF_FLOOR,
            "method": "EMERGENCY_FALLBACK",
            "source": "EMERGENCY_FALLBACK",
            "skip": False,
            "weights": weights,
            "detail": f"{self.loss.badge()} calculator unavailable; forced no-skip fallback",
        }

    def publish(self, next_issue, decision, votes):
        self.loss.touch()
        side = decision.get("side")
        if side not in ("B", "S"):
            side = list(self.history)[-1] if self.history else "B"
        prediction = letter_to_label(side)
        confidence = int(decision.get("conf", CONF_FLOOR))
        method = decision.get("method", "PATTERN")
        source = decision.get("source") or method
        self.pending_final = {
            "side": side,
            "period": next_issue,
            "conf": confidence,
            "method": method,
            "source": source,
        }

        win_rate = round(self.wins_final / self.total_final * 100, 1) if self.total_final else 0.0
        timestamp = ist_now().strftime("%Y-%m-%d %H:%M:%S")
        if self.loss.consec_losses == 0:
            level_desc = "NORMAL"
        elif self.loss.is_at_or_beyond_old_cap():
            level_desc = f"DEEP_RECOVER(L{self.loss.level})"
        else:
            level_desc = "RECOVER"

        self.latest = {
            "stream": STREAM_ID,
            "server": "DRAGO_MAIN",
            "target": "size",
            "version": "v36-recovery4-deep3-l7guard-noskip",
            "period": next_issue,
            "prediction": prediction,
            "confidence": confidence,
            "level": self.loss.level,
            "badge": self.loss.badge(),
            "consec_losses": self.loss.consec_losses,
            "level_desc": level_desc,
            "max_level_today": self.loss.max_level_today,
            "max_level_today_badge": self.loss.max_badge(),
            "max_consec_today": self.loss.max_consec_today,
            "stop_loss_hit": self.loss.stop_loss_hit,
            "stop_loss_streak": STOP_LOSS_STREAK,
            "level_cap": LEVEL_CAP,
            "method": method,
            "source": source,
            "status": "READY",
            "win_rate": win_rate,
            "wins": self.wins_final,
            "total": self.total_final,
            "draw_count": self.draw_count,
            "history_len": len(self.history),
            "weights": decision.get("weights") or {},
            "calc_votes": {name: (value["side"] if value else None) for name, value in votes.items()},
            "detail": decision.get("detail", ""),
            "learning_status": pattern.learning_status(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "timestamp": timestamp,
        }
        slim = {
            "server": "DRAGO_MAIN",
            "version": "v36-recovery4-deep3-l7guard-noskip",
            "period": next_issue,
            "prediction": prediction,
            "confidence": confidence,
            "level": self.loss.level,
            "badge": self.loss.badge(),
            "consec_losses": self.loss.consec_losses,
            "level_desc": level_desc,
            "max_level_today": self.loss.max_level_today,
            "max_level_today_badge": self.loss.max_badge(),
            "max_consec_today": self.loss.max_consec_today,
            "stop_loss_hit": self.loss.stop_loss_hit,
            "source": source,
            "play_time": "N/A",
            "timestamp": timestamp,
        }
        fb_put(FIREBASE_AUTOBET_URL, f"predictions/{STREAM_ID}", slim, FIREBASE_AUTOBET_SECRET)
        fb_put(FIREBASE_URL, f"predictions/{STREAM_ID}", self.latest, FIREBASE_SECRET)
        log.info(
            "PUBLISH %s -> %s conf=%s %s | level=%s consec=%s",
            next_issue, prediction, confidence, method, self.loss.badge(), self.loss.consec_losses,
        )

    def on_draw(self, period, number, learn_only=False):
        actual = number_to_size_letter(number)
        self.settle(actual)
        self.history.append(actual)
        self.history_nums.append(int(number))
        self.last_period = str(period).strip()
        self.draw_count += 1
        record_draw_row(period, number)
        if learn_only:
            pattern.predict(
                list(self.history), event_id=self.last_period, save=False,
                numbers=list(self.history_nums),
            )
            self.pending_calc = {}
            self.pending_final = None
            return
        votes = self.run_calcs()
        self.publish(next_period(self.last_period), self.decide(votes), votes)


ENGINE = Engine()

DATA_SOURCE = {
    "ok": None,
    "http_status": None,
    "error": "",
    "consecutive_failures": 0,
    "cloudflare_blocked": False,
    "last_success_at": "",
    "last_attempt_at": "",
}


def _issue_key(row):
    if not isinstance(row, dict):
        return None
    value = (
        row.get("issueNumber") or row.get("issue_number") or row.get("issue")
        or row.get("period") or row.get("issueNo")
    )
    return str(value).strip() if value is not None else None


def load_draw_store():
    global _draw_store
    with _draw_store_lock:
        if _draw_store:
            return list(_draw_store)
        try:
            if os.path.exists(DRAW_STORE_FILE):
                with open(DRAW_STORE_FILE, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                _draw_store = data if isinstance(data, list) else data.get("items", [])
            else:
                _draw_store = []
        except Exception as exc:
            log.warning("draw store load: %s", exc)
            _draw_store = []
        return list(_draw_store)


def save_draw_store():
    with _draw_store_lock:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(_draw_store),
            "items": _draw_store,
        }
        temporary = DRAW_STORE_FILE + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary, DRAW_STORE_FILE)


def record_draw_row(period, number):
    global _draw_store
    load_draw_store()
    try:
        number = int(number)
    except Exception:
        return
    period = str(period).strip()
    row = {
        "issueNumber": period,
        "number": str(number),
        "color": "red" if number in (1, 3, 5, 7, 9) else (
            "green" if number in (2, 4, 6, 8) else "violet"
        ),
        "size": "big" if number >= 5 else "small",
    }
    with _draw_store_lock:
        _draw_store = [item for item in _draw_store if _issue_key(item) != period]
        _draw_store.insert(0, row)
        del _draw_store[DRAW_STORE_MAX:]
    try:
        save_draw_store()
    except Exception as exc:
        log.warning("draw store save: %s", exc)


def require_vps_auth(request):
    if not VPS_SECRET:
        raise HTTPException(status_code=503, detail="VPS_SECRET not configured")
    key = (request.headers.get("x-vps-key") or "").strip()
    if not key:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    if key != VPS_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _update_data_source(ok, http_status=None, error=""):
    DATA_SOURCE.update(
        {
            "ok": ok,
            "http_status": http_status,
            "error": error,
            "last_attempt_at": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if ok:
        DATA_SOURCE["consecutive_failures"] = 0
        DATA_SOURCE["cloudflare_blocked"] = False
        DATA_SOURCE["last_success_at"] = DATA_SOURCE["last_attempt_at"]
    else:
        DATA_SOURCE["consecutive_failures"] += 1
        DATA_SOURCE["cloudflare_blocked"] = http_status == 403


def fetch_live_api():
    try:
        response = requests.get(
            API_URL,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://draw.ar-lottery01.com/",
                "Origin": "https://draw.ar-lottery01.com",
                "Cache-Control": "no-cache",
            },
        )
        if response.status_code != 200:
            _update_data_source(False, response.status_code, f"HTTP {response.status_code}")
            return []
        try:
            payload = response.json()
        except Exception:
            _update_data_source(False, response.status_code, "JSON parse error")
            return []
        records = []
        for item in (payload.get("data") or {}).get("list") or []:
            period, number = str(item.get("issueNumber", "")).strip(), item.get("number")
            try:
                number = int(number)
            except Exception:
                continue
            if period and 0 <= number <= 9:
                records.append({"period": period, "number": number})
        records.sort(key=lambda item: period_sort_key(item["period"]))
        _update_data_source(True, 200, "")
        return records
    except Exception as exc:
        _update_data_source(False, None, str(exc)[:120])
        log.warning("live fetch error: %s", exc)
        return []


def bootstrap_from_api():
    records = fetch_live_api()
    if not records:
        log.warning("bootstrap: no records; source=%s", DATA_SOURCE)
        return False
    for record in records:
        ENGINE.on_draw(record["period"], record["number"], learn_only=True)
    ENGINE.bootstrapped = True
    votes = ENGINE.run_calcs()
    ENGINE.publish(next_period(ENGINE.last_period), ENGINE.decide(votes), votes)
    log.info("bootstrap OK draws=%s last=%s", ENGINE.draw_count, ENGINE.last_period)
    return True


def tick():
    if not ENGINE.bootstrapped:
        bootstrap_from_api()
        return
    records = fetch_live_api()
    newer = [
        item for item in records
        if not ENGINE.last_period
        or period_sort_key(item["period"]) > period_sort_key(ENGINE.last_period)
    ]
    if not newer:
        return
    newer.sort(key=lambda item: period_sort_key(item["period"]))
    expected = next_period(ENGINE.last_period)
    if newer[0]["period"] != expected and ENGINE.pending_final:
        log.warning("period gap; pending bet %s voided", ENGINE.pending_final.get("period"))
        ENGINE.pending_final = None
        ENGINE.pending_calc = {}
    for record in newer[:-1]:
        ENGINE.on_draw(record["period"], record["number"], learn_only=True)
    live = newer[-1]
    ENGINE.on_draw(live["period"], live["number"], learn_only=False)


def poll_loop():
    while True:
        try:
            tick()
        except Exception as exc:
            log.exception("tick error: %s", exc)
        time.sleep(POLL_SEC)


def render_keepalive_loop():
    while True:
        try:
            requests.get(
                RENDER_KEEPALIVE_URL, timeout=15,
                headers={"User-Agent": "DRAGO-Orihost-KeepAlive/2.0"},
            )
        except Exception as exc:
            log.warning("Render keep-alive failed: %s", exc)
        time.sleep(max(60, RENDER_KEEPALIVE_SEC))


def midnight_cleanup_loop():
    ist = timezone(timedelta(hours=5, minutes=30))
    while True:
        try:
            now = datetime.now(ist)
            tomorrow = (now + timedelta(days=1)).date()
            target = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=ist)
            time.sleep(max((target - now).total_seconds(), 1))
            fb_delete(FIREBASE_AUTOBET_URL, "predictions", FIREBASE_AUTOBET_SECRET)
            log.info("autobet predictions wiped at IST midnight")
        except Exception as exc:
            log.error("midnight cleanup: %s", exc)
            time.sleep(60)


@asynccontextmanager
async def lifespan(app):
    load_draw_store()
    ENGINE.loss.load_state()
    threads = (
        (poll_loop, "poll-30s"),
        (render_keepalive_loop, "render-keepalive"),
        (midnight_cleanup_loop, "midnight"),
        (daily_report_loop, "daily-report"),
    )
    for target, name in threads:
        threading.Thread(target=target, daemon=True, name=name).start()
    telegram_send(
        "\n".join(
            [
                "🐉 DRAGO SERVER ONLINE",
                "━━━━━━━━━━━━━━━━━━",
                "Version: v36 Recovery4/Deep3/L7Guard No-SKIP",
                "Prediction: BIG/SMALL every valid round",
                "Level system: ACTIVE",
                f"Level cap: L{LEVEL_CAP}" if LEVEL_CAP else "Level cap: UNLIMITED",
                f"Stop-loss signal: {STOP_LOSS_STREAK} losses" if STOP_LOSS_STREAK else "Stop-loss: disabled",
                f"High-level alert: L{TELEGRAM_LEVEL_ALERT_AT}+",
                f"Daily report: {DAILY_REPORT_HOUR:02d}:00 IST",
                f"Started: {ist_now().strftime('%d-%m-%Y %H:%M:%S')} IST",
            ]
        ),
        silent=True,
    )
    log.info("DRAGO v36 started env=%s VPS_SECRET=%s", _ENV_FILE, "SET" if VPS_SECRET else "MISSING")
    yield


app = FastAPI(title="DRAGO Orihost v36", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "app": "DRAGO Main",
        "version": "v36-recovery4-deep3-l7guard-noskip",
        "stream": STREAM_ID,
        "mode": "Recovery8 No-SKIP; level system active",
        "methods": pattern.learning_status().get("active_methods", []),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "v36-recovery4-deep3-l7guard-noskip",
        "bootstrapped": ENGINE.bootstrapped,
        "draws": ENGINE.draw_count,
        "history_len": len(ENGINE.history),
        "last_period": ENGINE.last_period,
        "level": ENGINE.loss.level,
        "badge": ENGINE.loss.badge(),
        "consec_losses": ENGINE.loss.consec_losses,
        "max_level_today": ENGINE.loss.max_level_today,
        "max_consec_today": ENGINE.loss.max_consec_today,
        "level_cap": LEVEL_CAP or "unlimited",
        "stop_loss_streak": STOP_LOSS_STREAK or "disabled",
        "stop_loss_hit": ENGINE.loss.stop_loss_hit,
        "learning": pattern.learning_status(),
        "data_source": dict(DATA_SOURCE),
    }


@app.get("/api/prediction/{game}/{tf}/{target}")
def get_prediction(game: str, tf: str, target: str, request: Request):
    require_vps_auth(request)
    if target.lower() != "size":
        raise HTTPException(400, "only size supported")
    if game.lower() != "wingo" or tf.lower() not in ("30s", "30"):
        raise HTTPException(404, "only wingo/30s")
    return ENGINE.latest


@app.get("/api/history")
def get_history(request: Request, limit: int = 100000):
    require_vps_auth(request)
    limit = max(1, min(int(limit or 100000), DRAW_STORE_MAX))
    items = load_draw_store()[:limit]
    return {
        "success": True,
        "count": len(items),
        "limit": limit,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }


@app.get("/api/debug")
def debug(request: Request):
    require_vps_auth(request)
    ratio = ENGINE._regime_b_ratio()
    return {
        "latest": ENGINE.latest,
        "level": ENGINE.loss.level,
        "badge": ENGINE.loss.badge(),
        "consec_losses": ENGINE.loss.consec_losses,
        "weights": ENGINE.all_weights(),
        "last_history": list(ENGINE.history)[-40:],
        "regime_b_ratio": round(ratio, 3) if ratio is not None else None,
        "last_period": ENGINE.last_period,
        "learning_status": pattern.learning_status(),
        "top_patterns": pattern._state.top_patterns(10),
    }


@app.get("/api/mistakes/recent")
def mistakes_recent(request: Request, limit: int = 20):
    require_vps_auth(request)
    limit = max(1, min(int(limit), 200))
    if not os.path.exists(MISTAKES_FILE):
        return {"count": 0, "items": []}
    with _mistakes_lock:
        with open(MISTAKES_FILE, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    items = []
    for line in lines[-limit:]:
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    return {"count": len(items), "items": items}


@app.get("/api/mistakes/stats")
def mistakes_stats(request: Request):
    require_vps_auth(request)
    by_method, by_actual, total = defaultdict(int), defaultdict(int), 0
    if os.path.exists(MISTAKES_FILE):
        with _mistakes_lock:
            with open(MISTAKES_FILE, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    total += 1
                    by_method[item.get("method") or "?"] += 1
                    by_actual[item.get("actual") or item.get("actual_side") or "?"] += 1
    return {
        "total_losses_logged": total,
        "by_method": dict(by_method),
        "by_actual_side": dict(by_actual),
        "file": MISTAKES_FILE,
    }


@app.get("/api/patterns/top")
def top_patterns(request: Request, limit: int = 20):
    require_vps_auth(request)
    limit = max(1, min(int(limit), 100))
    return {
        "total_patterns": pattern._state.total_patterns(),
        "reliable_patterns": pattern._state.reliable_patterns(),
        "top": pattern._state.top_patterns(limit),
    }


def _logs_auth_ok(request, key=""):
    candidates = []
    if str(key).strip():
        candidates.append(str(key).strip())
    header_key = (request.headers.get("x-vps-key") or "").strip()
    if header_key:
        candidates.append(header_key)
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        candidates.append(auth[7:].strip())
    return any(
        candidate and ((VPS_SECRET and candidate == VPS_SECRET) or (LOGS_SECRET and candidate == LOGS_SECRET))
        for candidate in candidates
    )


@app.get("/api/mistakes/logs")
def mistakes_logs(request: Request, limit: int = 2000, key: str = ""):
    if not _logs_auth_ok(request, key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    limit = max(1, min(int(limit or 2000), 20000))
    if not os.path.exists(MISTAKES_FILE):
        return {"success": True, "count": 0, "total_in_file": 0, "items": []}
    with _mistakes_lock:
        with open(MISTAKES_FILE, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    items = []
    for line in lines[-limit:]:
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    items.reverse()
    return {
        "success": True,
        "file": MISTAKES_FILE,
        "total_in_file": len(lines),
        "count": len(items),
        "limit": limit,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }


if __name__ == "__main__":
    log.info("starting DRAGO v36 on port %s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
