#!/usr/bin/env python3
"""Walk-forward online backtest for DRAGO AI.

No future leakage:
- rows are sorted chronological
- prediction for row i uses rows < i only
- after result is known, model learns from that one result
"""

import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pattern

API_URL = os.getenv(
    "HISTORY_API_FULL_URL",
    "https://dragopredictor.onrender.com/v1/wingo30s/history?api_key=drago_d5b31311d951cec50aa1894ef614ebc73c039e0bacaf44c8&limit=10000",
)
OUT_JSON = os.getenv("BACKTEST_OUT_JSON", "ai_backtest_10k_stats.json")
OUT_MD = os.getenv("BACKTEST_OUT_MD", "AI_BACKTEST_10K_REPORT.md")
CACHE_JSON = os.getenv("BACKTEST_CACHE_JSON", "backtest_history_cache.json")
WARMUP = int(os.getenv("BACKTEST_WARMUP", "20") or 20)
RESET_MODEL = True


def _records_from_payload(payload):
    items = payload.get("items", payload if isinstance(payload, list) else [])
    return pattern.normalise_records(items)


def fetch_records():
    force_fetch = (os.getenv("BACKTEST_FORCE_FETCH", "0") or "0").lower() in ("1", "true", "yes", "y")
    if not force_fetch and os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _records_from_payload(payload), payload
    try:
        with urlopen(API_URL, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        # Save raw payload so repeated optimization does not burn API quota.
        with open(CACHE_JSON, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return _records_from_payload(payload), payload
    except (HTTPError, URLError) as exc:
        if os.path.exists(CACHE_JSON):
            with open(CACHE_JSON, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            print(f"API fetch failed ({exc}); using local cache {CACHE_JSON}")
            return _records_from_payload(payload), payload
        raise RuntimeError(
            f"History API fetch failed and no cache exists: {exc}. "
            f"Run again when API quota is available or place raw API JSON at {CACHE_JSON}."
        )


def pct(x, total):
    return round((x / total * 100.0), 2) if total else 0.0


def conf_bucket(conf):
    conf = int(conf)
    if conf < 56:
        return "54-55"
    if conf < 58:
        return "56-57"
    if conf < 60:
        return "58-59"
    if conf < 62:
        return "60-61"
    if conf < 64:
        return "62-63"
    if conf < 66:
        return "64-65"
    return "66+"


def run_backtest(records):
    ai = pattern.OnlineAIPredictor()
    # Important: do not load/write production state; pure fresh walk-forward model.
    ai.save_state = lambda: None
    history, nums, colors = [], [], []
    bets = wins = losses = 0
    consec_loss = 0
    consec_win = 0
    max_consec_loss = 0
    max_consec_win = 0
    max_level = 1
    level_attempts = Counter()
    level_wins = Counter()
    level_losses = Counter()
    level_first_seen_period = {}
    loss_streak_hist = Counter()
    win_streak_hist = Counter()
    pred_counts = Counter()
    actual_counts = Counter()
    method_counts = Counter()
    method_perf = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0})
    conf_perf = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0})
    hour_perf = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "max_cl": 0})
    number_perf = defaultdict(lambda: {"actual": 0, "pred_win": 0, "pred_loss": 0})
    color_perf = defaultdict(lambda: {"actual": 0, "pred_win": 0, "pred_loss": 0})
    side_matrix = defaultdict(Counter)
    results = []

    current_loss_run = 0
    current_win_run = 0

    start = time.time()
    for idx, row in enumerate(records):
        actual = row["side"]
        number = row["number"]
        color = row["color"]
        actual_counts[actual] += 1
        number_perf[str(number)]["actual"] += 1
        color_perf[color]["actual"] += 1

        # First rows only build context. This is not counted as bet.
        if len(history) < WARMUP:
            history.append(actual)
            nums.append(number)
            colors.append(color)
            continue

        level = consec_loss + 1
        level_attempts[level] += 1
        level_first_seen_period.setdefault(level, row["period"])
        max_level = max(max_level, level)

        pred = ai.predict(
            history,
            event_id=row["period"],
            numbers=nums,
            colors=colors,
            consec_losses=consec_loss,
            save=True,
        )
        side = pred.get("side") if pred.get("side") in ("B", "S") else "B"
        meta = pred.get("meta") or {}
        method = meta.get("source") or "AI_NN"
        confidence = int(pred.get("conf", 0) or 0)
        correct = side == actual

        bets += 1
        pred_counts[side] += 1
        method_counts[method] += 1
        method_perf[method]["bets"] += 1
        conf_key = conf_bucket(confidence)
        conf_perf[conf_key]["bets"] += 1
        side_matrix[side][actual] += 1

        try:
            seq = int(str(row["period"])[-4:])
            hour = max(0, min(23, ((seq - 1) * 30) // 3600))
        except Exception:
            hour = 0
        hour_perf[hour]["bets"] += 1

        if correct:
            wins += 1
            level_wins[level] += 1
            method_perf[method]["wins"] += 1
            conf_perf[conf_key]["wins"] += 1
            hour_perf[hour]["wins"] += 1
            number_perf[str(number)]["pred_win"] += 1
            color_perf[color]["pred_win"] += 1
            current_win_run += 1
            if current_loss_run:
                loss_streak_hist[current_loss_run] += 1
            current_loss_run = 0
            consec_loss = 0
        else:
            losses += 1
            level_losses[level] += 1
            method_perf[method]["losses"] += 1
            conf_perf[conf_key]["losses"] += 1
            hour_perf[hour]["losses"] += 1
            number_perf[str(number)]["pred_loss"] += 1
            color_perf[color]["pred_loss"] += 1
            if current_win_run:
                win_streak_hist[current_win_run] += 1
            current_win_run = 0
            current_loss_run += 1
            consec_loss += 1
            max_consec_loss = max(max_consec_loss, consec_loss)
            max_level = max(max_level, consec_loss + 1)

        max_consec_win = max(max_consec_win, current_win_run)
        hour_perf[hour]["max_cl"] = max(hour_perf[hour]["max_cl"], consec_loss)

        loss_boost = 1 if correct else min(10, max(3, (level - 1) + 3))
        learn_report = ai.learn_from_actual(
            actual,
            actual_number=number,
            actual_color=color,
            period=row["period"],
            loss_boost=loss_boost,
        )

        results.append(
            {
                "idx": idx,
                "period": row["period"],
                "level": level,
                "prediction": side,
                "actual": actual,
                "number": number,
                "color": color,
                "confidence": confidence,
                "method": method,
                "correct": correct,
                "consec_loss_after": consec_loss,
                "consec_win_after": current_win_run,
                "prob_big": meta.get("prob_big"),
                "loss_after_train": learn_report.get("loss_after") if isinstance(learn_report, dict) else None,
            }
        )

        history.append(actual)
        nums.append(number)
        colors.append(color)

    if current_loss_run:
        loss_streak_hist[current_loss_run] += 1
    if current_win_run:
        win_streak_hist[current_win_run] += 1

    # Convert counters to sorted serializable dicts.
    def sorted_counter(counter):
        return {str(k): int(counter[k]) for k in sorted(counter)}

    def perf_dict(perf):
        out = {}
        for k in sorted(perf, key=lambda x: str(x)):
            v = perf[k]
            b = int(v.get("bets", 0) or 0)
            w = int(v.get("wins", 0) or 0)
            l = int(v.get("losses", 0) or 0)
            out[str(k)] = {"bets": b, "wins": w, "losses": l, "wr_pct": pct(w, b)}
            if "max_cl" in v:
                out[str(k)]["max_cl"] = int(v.get("max_cl", 0) or 0)
        return out

    level_perf = {}
    for level in sorted(level_attempts):
        b = level_attempts[level]
        w = level_wins[level]
        l = level_losses[level]
        level_perf[f"L{level}"] = {
            "attempts": int(b),
            "wins": int(w),
            "losses": int(l),
            "wr_pct": pct(w, b),
            "first_seen_period": level_first_seen_period.get(level),
        }

    conf_values = [r["confidence"] for r in results if r["confidence"]]
    stats = {
        "meta": {
            "mode": "fresh walk-forward online learning",
            "future_leakage": False,
            "warmup_rows_no_bet": WARMUP,
            "api_url": API_URL.replace("api_key=", "api_key=***"),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - start, 2),
        },
        "data": {
            "records": len(records),
            "first_period": records[0]["period"] if records else None,
            "last_period": records[-1]["period"] if records else None,
            "side_counts": dict(Counter(r["side"] for r in records)),
            "number_counts": {str(i): Counter(r["number"] for r in records)[i] for i in range(10)},
            "color_counts": dict(Counter(r["color"] for r in records)),
        },
        "overall": {
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": pct(wins, bets),
            "loss_rate_pct": pct(losses, bets),
            "max_consecutive_losses": max_consec_loss,
            "max_consecutive_wins": max_consec_win,
            "highest_level_reached": f"L{max_level}",
            "highest_level_number": max_level,
            "prediction_counts": dict(pred_counts),
            "actual_counts_all_rows": dict(actual_counts),
            "confidence_avg": round(statistics.mean(conf_values), 2) if conf_values else None,
            "confidence_min": min(conf_values) if conf_values else None,
            "confidence_max": max(conf_values) if conf_values else None,
        },
        "level_performance": level_perf,
        "loss_streak_distribution": sorted_counter(loss_streak_hist),
        "win_streak_distribution": sorted_counter(win_streak_hist),
        "method_performance": perf_dict(method_perf),
        "confidence_performance": perf_dict(conf_perf),
        "hourly_performance": perf_dict(hour_perf),
        "number_performance": {
            str(k): {
                "actual_count": int(v["actual"]),
                "wins_when_actual": int(v["pred_win"]),
                "losses_when_actual": int(v["pred_loss"]),
                "wr_pct_when_actual": pct(v["pred_win"], v["pred_win"] + v["pred_loss"]),
            }
            for k, v in sorted(number_perf.items(), key=lambda kv: int(kv[0]))
        },
        "color_performance": {
            str(k): {
                "actual_count": int(v["actual"]),
                "wins_when_actual": int(v["pred_win"]),
                "losses_when_actual": int(v["pred_loss"]),
                "wr_pct_when_actual": pct(v["pred_win"], v["pred_win"] + v["pred_loss"]),
            }
            for k, v in sorted(color_perf.items())
        },
        "prediction_actual_matrix": {pred: dict(actuals) for pred, actuals in side_matrix.items()},
        "ai_status_end": ai.learning_status(compact=True),
        "last_50_results": results[-50:],
    }
    return stats


def write_report(stats):
    overall = stats["overall"]
    lines = []
    lines.append(f"# DRAGO AI 10K Walk-Forward Backtest — {pattern.VERSION}")
    lines.append("")
    lines.append("Mode: fresh online learning. Prediction uses only previous rows; after result, AI learns from that result.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Records: `{stats['data']['records']}`")
    lines.append(f"- Betting rows after warmup: `{overall['bets']}`")
    lines.append(f"- Wins: `{overall['wins']}`")
    lines.append(f"- Losses: `{overall['losses']}`")
    lines.append(f"- Win rate: `{overall['win_rate_pct']}%`")
    lines.append(f"- Max consecutive losses: `{overall['max_consecutive_losses']}`")
    lines.append(f"- Max consecutive wins: `{overall['max_consecutive_wins']}`")
    lines.append(f"- Highest level reached: `{overall['highest_level_reached']}`")
    lines.append(f"- Prediction counts: `{overall['prediction_counts']}`")
    lines.append(f"- Average confidence: `{overall['confidence_avg']}`")
    lines.append("")
    lines.append("## Level performance")
    lines.append("")
    lines.append("| Level | Attempts | Wins | Losses | WR % | First seen period |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for level, row in sorted(stats["level_performance"].items(), key=lambda kv: int(kv[0][1:])):
        lines.append(f"| {level} | {row['attempts']} | {row['wins']} | {row['losses']} | {row['wr_pct']} | {row.get('first_seen_period') or ''} |")
    lines.append("")
    lines.append("## Loss streak distribution")
    lines.append("")
    lines.append("| Loss streak length | Count |")
    lines.append("|---:|---:|")
    for k, v in stats["loss_streak_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Win streak distribution")
    lines.append("")
    lines.append("| Win streak length | Count |")
    lines.append("|---:|---:|")
    for k, v in stats["win_streak_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Confidence performance")
    lines.append("")
    lines.append("| Confidence bucket | Bets | Wins | Losses | WR % |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, row in stats["confidence_performance"].items():
        lines.append(f"| {k} | {row['bets']} | {row['wins']} | {row['losses']} | {row['wr_pct']} |")
    lines.append("")
    lines.append("## Number performance, based on actual number")
    lines.append("")
    lines.append("| Number | Actual count | Prediction wins | Prediction losses | WR % |")
    lines.append("|---:|---:|---:|---:|---:|")
    for k, row in stats["number_performance"].items():
        lines.append(f"| {k} | {row['actual_count']} | {row['wins_when_actual']} | {row['losses_when_actual']} | {row['wr_pct_when_actual']} |")
    lines.append("")
    lines.append("## Colour performance, based on actual colour")
    lines.append("")
    lines.append("| Colour | Actual count | Prediction wins | Prediction losses | WR % |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, row in stats["color_performance"].items():
        lines.append(f"| {k} | {row['actual_count']} | {row['wins_when_actual']} | {row['losses_when_actual']} | {row['wr_pct_when_actual']} |")
    lines.append("")
    lines.append("## Hourly performance")
    lines.append("")
    lines.append("| Hour | Bets | Wins | Losses | WR % | Max CL |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for k, row in sorted(stats["hourly_performance"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {k} | {row['bets']} | {row['wins']} | {row['losses']} | {row['wr_pct']} | {row.get('max_cl', 0)} |")
    lines.append("")
    lines.append("## End AI status")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats["ai_status_end"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Note: Data is close to random. Backtest is for behaviour/statistics, not a guarantee of future profit.")
    return "\n".join(lines) + "\n"


def main():
    records, payload = fetch_records()
    print(f"Fetched {len(records)} records")
    stats = run_backtest(records)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    report = write_report(stats)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(json.dumps(stats["overall"], indent=2))
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
