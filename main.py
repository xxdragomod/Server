# main.py — DRAGO AI WinGo 30s Server
# Old rule/guard prediction system removed. Level system remains active.
# New prediction engine: pattern.py online neural AI trained on number + size + colour.

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse


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
            if key:
                # Keep current behaviour: local .env is authoritative for this project.
                os.environ[key] = value
        return str(env_path)
    return None


_ENV_FILE = load_dotenv_file()

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import pattern

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drago")

STREAM_ID = "wingo_30s"
VERSION = "v48-ai-v7-live-learning"

# History source.  Prefer old host env names too, so deployments that already
# had API_URL/SOURCE_API keep working "jaise pehle tha".
DEFAULT_DRAGO_HISTORY_API_URL = "https://dragopredictor.onrender.com/v1/wingo30s/history"
DEFAULT_DIRECT_HISTORY_API_URL = "https://draw.ar-lottery02.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HISTORY_API_URL = (
    os.getenv("HISTORY_API_URL")
    or os.getenv("API_URL")
    or os.getenv("SOURCE_API")
    or DEFAULT_DRAGO_HISTORY_API_URL
).strip()
HISTORY_FALLBACK_URLS = (os.getenv("HISTORY_FALLBACK_URLS") or DEFAULT_DIRECT_HISTORY_API_URL).strip()
HISTORY_API_KEY = os.getenv(
    "HISTORY_API_KEY",
    "drago_d5b31311d951cec50aa1894ef614ebc73c039e0bacaf44c8",
).strip()
BOOTSTRAP_HISTORY_LIMIT = int(os.getenv("BOOTSTRAP_HISTORY_LIMIT", "10000") or 10000)
POLL_HISTORY_LIMIT = int(os.getenv("POLL_HISTORY_LIMIT", "200") or 200)
POLL_SEC = int(os.getenv("POLL_SEC", "3") or 3)
HISTORY_MAX = int(os.getenv("HISTORY_MAX", "5000") or 5000)
DRAW_STORE_MAX = int(os.getenv("DRAW_STORE_MAX", "100000") or 100000)
# Old live server used wingo30s.json with root data/savedAt and up to 1 lakh rows.
# Keep that behaviour, while also mirroring the newer wingo30s_history.json name.
DRAW_STORE_FILE = os.getenv("DRAW_STORE_FILE", "wingo30s.json")
_DRAW_STORE_DEFAULT_MIRROR = "wingo30s_history.json" if DRAW_STORE_FILE != "wingo30s_history.json" else "wingo30s.json"
DRAW_STORE_MIRROR_FILES = os.getenv("DRAW_STORE_MIRROR_FILES", _DRAW_STORE_DEFAULT_MIRROR)
AI_ANALYSIS_FILE = os.getenv("AI_ANALYSIS_FILE", "ai_analysis.json")
# V7 trains a recent live window at startup (fast) and then keeps learning every
# settled round. Full 10K training is too slow on small hosts, so keep a cap.
AI_BOOTSTRAP_EPOCHS = int(os.getenv("AI_BOOTSTRAP_EPOCHS", "1") or 1)
AI_BOOTSTRAP_TRAIN_LIMIT = int(os.getenv("AI_BOOTSTRAP_TRAIN_LIMIT", "1500") or 1500)
AI_RESET_ON_BOOT = (os.getenv("AI_RESET_ON_BOOT", "0") or "0").strip().lower() in ("1", "true", "yes", "y")
HISTORY_API_MIN_INTERVAL_SEC = int(os.getenv("HISTORY_API_MIN_INTERVAL_SEC", "28") or 28)
HISTORY_API_BACKOFF_SEC = int(os.getenv("HISTORY_API_BACKOFF_SEC", "300") or 300)
# Fake/clock fallback is OFF by default. A new prediction is published only when
# a real fresh draw is received from the history source. Set CLOCK_FALLBACK_ENABLE=1
# only for UI demo mode; do not use it for real/autobet play.
# Hard-disabled: fallback without real result made repeated SMALL/L2 predictions.
CLOCK_FALLBACK_ENABLE = False
CLOCK_FALLBACK_AFTER_SEC = int(os.getenv("CLOCK_FALLBACK_AFTER_SEC", "45") or 45)

FIREBASE_AUTOBET_URL = os.getenv(
    "FIREBASE_AUTOBET_URL", "https://auto-bet-pro-default-rtdb.firebaseio.com"
)
FIREBASE_AUTOBET_SECRET = os.getenv("FIREBASE_AUTOBET_SECRET", "")
FIREBASE_URL = os.getenv(
    "FIREBASE_URL", "https://drago-predictor-default-rtdb.firebaseio.com"
)
FIREBASE_SECRET = os.getenv("FIREBASE_SECRET", "")

PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "30252")))
VPS_SECRET = (os.getenv("VPS_SECRET") or os.getenv("DRAGO_VPS_SECRET") or "").strip()
LOGS_SECRET = (os.getenv("LOGS_SECRET") or os.getenv("MISTAKES_LOG_KEY") or "").strip()

# Existing hardcoded Telegram fallback retained as requested.
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or "8782822978:AAEitqI-CdxbiAN3-55Ltf72i79BpIjBeaA"
).strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "6656009938").strip()
TELEGRAM_LEVEL_ALERT_AT = int(os.getenv("TELEGRAM_LEVEL_ALERT_AT", "5") or 5)
DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR", "21") or 21)
SESSION_END_HOUR = int(os.getenv("SESSION_END_HOUR", "21") or 21)
HOURLY_MIN_SAMPLE = int(os.getenv("HOURLY_MIN_SAMPLE", "10") or 10)

LEVEL_CAP = int(os.getenv("LEVEL_CAP", "0") or 0)
# Ultra-safe paper pause removed as requested. Server will keep real/autobet
# play_signal=PLAY; this value remains only for response compatibility.
ULTRA_SAFE_MODE = False
ULTRA_SAFE_PAUSE_AFTER_LOSSES = int(os.getenv("ULTRA_SAFE_PAUSE_AFTER_LOSSES", "5") or 5)
STOP_LOSS_STREAK = int(os.getenv("STOP_LOSS_STREAK", "9") or 9)
LEVEL_STATE_FILE = os.getenv("LEVEL_STATE_FILE", "level_state.json")
MISTAKES_FILE = os.getenv("MISTAKES_FILE", "mistakes.jsonl")
MISTAKES_MAX_LINES = int(os.getenv("MISTAKES_MAX_LINES", "1000") or 1000)

RENDER_KEEPALIVE_URL = (
    os.getenv("RENDER_KEEPALIVE_URL") or "https://dragopredictor.onrender.com"
).rstrip("/")
RENDER_KEEPALIVE_SEC = int(os.getenv("RENDER_KEEPALIVE_SEC", "180") or 180)

_level_lock = threading.Lock()
_mistakes_lock = threading.Lock()
_draw_store_lock = threading.Lock()
_hourly_lock = threading.Lock()
_draw_store = []
_hourly_stats = {}
_tg_last_level_alert = -1
_history_api_last_call_ts = 0.0
_history_api_backoff_until = 0.0

DATA_SOURCE = {
    "ok": None,
    "http_status": None,
    "error": "",
    "consecutive_failures": 0,
    "last_success_at": "",
    "last_attempt_at": "",
    "source": HISTORY_API_URL,
}


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


def number_to_size_letter(number):
    return "B" if int(number) >= 5 else "S"


def letter_to_label(letter):
    return {"B": "BIG", "S": "SMALL"}.get(letter, "WAIT")


def color_from_number(number):
    return pattern.infer_color(number)


def period_sort_key(period):
    return pattern.period_sort_key(period)


def next_period(period):
    text = str(period).strip()
    if not text.isdigit() or len(text) != 17:
        return str(int(text) + 1) if text.isdigit() else text
    date_part, middle, sequence = text[:8], text[8:13], int(text[13:])
    if sequence < 2880:
        return f"{date_part}{middle}{sequence + 1:04d}"
    next_day = datetime.strptime(date_part, "%Y%m%d") + timedelta(days=1)
    return f"{next_day.strftime('%Y%m%d')}{middle}0001"


def telegram_send(text, silent=False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("telegram not configured")
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
    global _tg_last_level_alert
    if level < TELEGRAM_LEVEL_ALERT_AT or _tg_last_level_alert >= 0:
        return
    _tg_last_level_alert = consec
    telegram_send(
        "\n".join(
            [
                "🚨 DRAGO AI HIGH-LEVEL ALERT",
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
                "AI model ne is loss context ko extra training weight se learn kiya.",
            ]
        )
    )


def telegram_recovery_alert(consec_before, level_before, period, method):
    if level_before < TELEGRAM_LEVEL_ALERT_AT:
        return
    telegram_send(
        "\n".join(
            [
                "✅ DRAGO AI RECOVERY WIN",
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
    try:
        with _mistakes_lock:
            with open(MISTAKES_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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


def _issue_key(row):
    if not isinstance(row, dict):
        return None
    value = row.get("issueNumber") or row.get("issue_number") or row.get("issue") or row.get("period") or row.get("issueNo")
    return str(value).strip() if value is not None else None


def _draw_store_files():
    files = []
    for raw in [DRAW_STORE_FILE, *(DRAW_STORE_MIRROR_FILES.split(",") if DRAW_STORE_MIRROR_FILES else [])]:
        path = str(raw or "").strip()
        if path and path not in files:
            files.append(path)
    return files


def _extract_store_items(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("items", "data", "records", "history"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return list(value.values())
    # Firebase-style indexed object: {"1": {...}, "2": {...}}
    if payload and all(isinstance(v, dict) for v in payload.values()):
        return list(payload.values())
    return []


def _draw_row(period, number, color=None, saved_at=None):
    number = int(number)
    side = number_to_size_letter(number)
    saved_at = saved_at or datetime.now(timezone.utc).isoformat()
    return {
        "issueNumber": str(period).strip(),
        "number": str(number),
        "color": pattern.clean_color(color, number),
        "size": "big" if side == "B" else "small",
        "savedAt": saved_at,
    }


def _normalise_store_item(item, fallback_saved_at=None):
    if not isinstance(item, dict):
        return None
    period = _issue_key(item)
    number = item.get("number")
    if period is None or number is None:
        return None
    try:
        return _draw_row(
            period,
            number,
            item.get("color"),
            item.get("savedAt") or item.get("saved_at") or item.get("timestamp") or fallback_saved_at,
        )
    except Exception:
        return None


def _normalise_store_items(items, fallback_saved_at=None):
    dedup = {}
    for item in items or []:
        row = _normalise_store_item(item, fallback_saved_at=fallback_saved_at)
        if row:
            dedup[row["issueNumber"]] = row
    rows = list(dedup.values())
    rows.sort(key=lambda row: period_sort_key(row["issueNumber"]), reverse=True)
    return rows[:DRAW_STORE_MAX]


def _draw_store_payload(items=None):
    rows = list(_draw_store if items is None else items)[:DRAW_STORE_MAX]
    saved_at = datetime.now(timezone.utc).isoformat()
    return {
        "success": True,
        "savedAt": saved_at,
        "savedAtIst": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": saved_at,
        "count": len(rows),
        "limit": DRAW_STORE_MAX,
        # Keep both names: old clients used data, newer endpoint used items.
        "data": rows,
        "items": rows,
    }


def load_draw_store():
    global _draw_store
    with _draw_store_lock:
        if _draw_store:
            return list(_draw_store)
        for path in _draw_store_files():
            try:
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                fallback_saved_at = None
                if isinstance(data, dict):
                    fallback_saved_at = data.get("savedAt") or data.get("updated_at")
                _draw_store = _normalise_store_items(_extract_store_items(data), fallback_saved_at=fallback_saved_at)
                log.info("draw store loaded %s rows from %s", len(_draw_store), path)
                return list(_draw_store)
            except Exception as exc:
                log.warning("draw store load %s: %s", path, exc)
        _draw_store = []
        return []


def save_draw_store():
    with _draw_store_lock:
        _draw_store.sort(key=lambda row: period_sort_key(row.get("issueNumber")), reverse=True)
        del _draw_store[DRAW_STORE_MAX:]
        payload = _draw_store_payload(_draw_store)
        for path in _draw_store_files():
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
            os.replace(temporary, path)


def replace_draw_store(records):
    """Merge freshly fetched records into the 1-lakh local store; do not wipe old rows."""
    global _draw_store
    existing = load_draw_store()
    now_saved_at = datetime.now(timezone.utc).isoformat()
    rows = [_draw_row(r["period"], r["number"], r.get("color"), saved_at=now_saved_at) for r in records]
    dedup = {row["issueNumber"]: row for row in _normalise_store_items(existing)}
    for row in rows:
        dedup[row["issueNumber"]] = row
    merged = list(dedup.values())
    merged.sort(key=lambda row: period_sort_key(row["issueNumber"]), reverse=True)
    with _draw_store_lock:
        _draw_store = merged[:DRAW_STORE_MAX]
    try:
        save_draw_store()
    except Exception as exc:
        log.warning("draw store save: %s", exc)


def record_draw_row(period, number, color=None):
    global _draw_store
    load_draw_store()
    row = _draw_row(period, number, color)
    with _draw_store_lock:
        _draw_store = [item for item in _draw_store if _issue_key(item) != row["issueNumber"]]
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
        DATA_SOURCE["last_success_at"] = DATA_SOURCE["last_attempt_at"]
    else:
        DATA_SOURCE["consecutive_failures"] += 1


def _safe_url(url):
    try:
        parsed = urlparse(str(url))
        query = parse_qs(parsed.query)
        if "api_key" in query:
            return str(url).replace(query["api_key"][0], "***")
    except Exception:
        pass
    return str(url)


def _history_source_urls():
    urls = []
    for raw in [HISTORY_API_URL, *(HISTORY_FALLBACK_URLS.split(",") if HISTORY_FALLBACK_URLS else [])]:
        url = str(raw or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _api_params(url, limit):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    host = (parsed.netloc or "").lower()
    params = {}
    # DRAGO/custom APIs accept limit + api_key.  The direct WinGo endpoint usually
    # rejects unnecessary bot-looking params, so keep it clean.
    if "dragopredictor" in host or "api_key" in query or "limit" in query:
        params["limit"] = int(limit)
        if HISTORY_API_KEY and "api_key" not in query:
            params["api_key"] = HISTORY_API_KEY
    elif "ar-lottery" not in host:
        params["limit"] = int(limit)
    return params


def _api_headers(url):
    host = (urlparse(url).netloc or "").lower()
    if "ar-lottery" in host:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://draw.ar-lottery01.com/",
            "Origin": "https://draw.ar-lottery01.com",
            "Cache-Control": "no-cache",
        }
    return {"User-Agent": "DRAGO-AI-Server/4.0", "Accept": "application/json"}


def _extract_items(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    return (
        payload.get("items")
        or payload.get("list")
        or data.get("list")
        or data.get("items")
        or []
    )


def _fetch_one_history_source(url, limit):
    try:
        response = requests.get(
            url,
            params=_api_params(url, limit),
            timeout=25,
            headers=_api_headers(url),
        )
        if response.status_code != 200:
            body = ""
            try:
                body = response.text[:180]
            except Exception:
                pass
            return [], response.status_code, f"HTTP {response.status_code} {body}".strip(), {}
        try:
            payload = response.json()
        except Exception:
            return [], response.status_code, "JSON parse error", {}
        records = pattern.normalise_records(_extract_items(payload))
        return records, response.status_code, "", payload if isinstance(payload, dict) else {}
    except Exception as exc:
        return [], None, str(exc)[:180], {}


def fetch_history_api(limit=BOOTSTRAP_HISTORY_LIMIT, force=False):
    global _history_api_last_call_ts, _history_api_backoff_until
    now_ts = time.time()
    if not force:
        if now_ts < _history_api_backoff_until:
            DATA_SOURCE["throttled"] = True
            DATA_SOURCE["error"] = "history API backoff active"
            DATA_SOURCE["next_allowed_in_sec"] = round(_history_api_backoff_until - now_ts, 1)
            return None
        wait_left = HISTORY_API_MIN_INTERVAL_SEC - (now_ts - _history_api_last_call_ts)
        if wait_left > 0:
            DATA_SOURCE["throttled"] = True
            DATA_SOURCE["error"] = "waiting for next allowed history API poll"
            DATA_SOURCE["next_allowed_in_sec"] = round(wait_left, 1)
            return None
    _history_api_last_call_ts = now_ts
    DATA_SOURCE["throttled"] = False
    DATA_SOURCE["next_allowed_in_sec"] = 0

    attempts = []
    merged = {}
    freshest = None
    for url in _history_source_urls():
        records, status, error, payload = _fetch_one_history_source(url, limit)
        latest_period = records[-1]["period"] if records else ""
        safe = _safe_url(url)
        attempt = {
            "source": safe,
            "ok": bool(records),
            "http_status": status,
            "count": len(records),
            "latest_period": latest_period,
            "error": error[:180] if error else "",
        }
        attempts.append(attempt)
        for row in records:
            merged[row["period"]] = row
        if records and (freshest is None or period_sort_key(latest_period) > period_sort_key(freshest["latest_period"])):
            freshest = {
                "status": status,
                "payload": payload,
                "source": safe,
                "latest_period": latest_period,
            }

    DATA_SOURCE["attempts"] = attempts[-5:]
    DATA_SOURCE["sources"] = [_safe_url(u) for u in _history_source_urls()]
    if merged and freshest:
        combined = [merged[k] for k in sorted(merged, key=period_sort_key)]
        _update_data_source(True, freshest["status"], "")
        DATA_SOURCE["source"] = freshest["source"]
        DATA_SOURCE["count"] = len(combined)
        DATA_SOURCE["limit"] = int(limit)
        DATA_SOURCE["latest_period"] = combined[-1]["period"]
        DATA_SOURCE["api_updated_at"] = freshest["payload"].get("updated_at") if isinstance(freshest["payload"], dict) else ""
        return combined

    error_text = "; ".join(
        f"{a.get('source')} -> {a.get('error') or a.get('http_status')}" for a in attempts
    )[:240] or "no history source configured"
    last_status = attempts[-1].get("http_status") if attempts else None
    _update_data_source(False, last_status, error_text)
    bad_statuses = {a.get("http_status") for a in attempts}
    if bad_statuses.intersection({402, 403, 429, 500, 502, 503, 504}):
        _history_api_backoff_until = time.time() + HISTORY_API_BACKOFF_SEC
        backoff_until = ist_now() + timedelta(seconds=HISTORY_API_BACKOFF_SEC)
        DATA_SOURCE["backoff_until_ist"] = backoff_until.strftime("%Y-%m-%d %H:%M:%S")
        DATA_SOURCE["backoff_sec"] = HISTORY_API_BACKOFF_SEC
    log.warning("history sources failed: %s", error_text)
    return []


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.history = deque(maxlen=HISTORY_MAX)
        self.history_nums = deque(maxlen=HISTORY_MAX)
        self.history_colors = deque(maxlen=HISTORY_MAX)
        self.last_period = None
        self.bootstrapped = False
        self.pending_final = None
        self.loss = LossLevel()
        self.total_final = 0
        self.wins_final = 0
        self.draw_count = 0
        self.last_publish_wall_ts = 0.0
        self.clock_fallback_count = 0
        self.bootstrap_info = {}
        self.analysis = {}
        self.latest = {
            "stream": STREAM_ID,
            "period": "",
            "prediction": "WAIT",
            "confidence": 0,
            "level": 1,
            "badge": "L1",
            "consec_losses": 0,
            "method": "AI_INIT",
            "status": "WAIT",
            "updated_at": "",
        }

    def snapshot_history(self):
        with self.lock:
            return list(self.history), list(self.history_nums), list(self.history_colors)

    def load_records(self, records):
        records = pattern.normalise_records(records)
        with self.lock:
            self.history.clear()
            self.history_nums.clear()
            self.history_colors.clear()
            for row in records[-HISTORY_MAX:]:
                self.history.append(row["side"])
                self.history_nums.append(row["number"])
                self.history_colors.append(row["color"])
            self.last_period = records[-1]["period"] if records else None
            self.draw_count = len(records)
            self.bootstrapped = bool(records)
        replace_draw_store(records)

    def bootstrap_from_api(self):
        records = fetch_history_api(BOOTSTRAP_HISTORY_LIMIT, force=True)
        source_name = "history_api"
        if not records:
            # If the paid/free external history API is exhausted, do not keep the
            # whole server stuck. Restart from persisted local draw store instead.
            cached_items = load_draw_store()
            records = pattern.normalise_records(cached_items)
            source_name = "local_draw_store"
        if not records:
            log.warning("bootstrap: no records; source=%s", DATA_SOURCE)
            return False
        log.info("bootstrap: loaded %s records from %s", len(records), source_name)
        analysis = pattern.deep_analyze_records(records)
        if AI_BOOTSTRAP_EPOCHS > 0:
            train_records = records[-AI_BOOTSTRAP_TRAIN_LIMIT:] if AI_BOOTSTRAP_TRAIN_LIMIT > 0 else records
            log.info(
                "bootstrap: training AI epochs=%s samples_window=%s/%s...",
                AI_BOOTSTRAP_EPOCHS,
                len(train_records),
                len(records),
            )
            train_info = pattern.fit_history(train_records, epochs=AI_BOOTSTRAP_EPOCHS, reset=AI_RESET_ON_BOOT)
            train_info["source"] = source_name
            train_info["train_window_records"] = len(train_records)
            train_info["full_records_loaded"] = len(records)
        else:
            train_info = {
                "trained": True,
                "bootstrap_trained": False,
                "online_learning_active": True,
                "reason": "AI_BOOTSTRAP_EPOCHS=0; live online learning + embedded V7 learned policy active",
                "records": len(records),
                "source": source_name,
            }
        self.load_records(records)
        with self.lock:
            self.analysis = analysis
            self.bootstrap_info = train_info
        try:
            with open(AI_ANALYSIS_FILE, "w", encoding="utf-8") as handle:
                json.dump({"analysis": analysis, "training": train_info}, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            log.warning("analysis save failed: %s", exc)
        self.publish(next_period(self.last_period))
        log.info("bootstrap OK draws=%s last=%s train=%s", self.draw_count, self.last_period, train_info)
        return True

    def settle(self, period, actual, number, color):
        if not self.pending_final:
            return False
        pending = self.pending_final
        pending_period = str(pending.get("period") or "")
        period = str(period)
        if pending_period and period != pending_period:
            if period_sort_key(period) > period_sort_key(pending_period):
                log.warning("period gap; pending AI bet %s voided by %s", pending_period, period)
                self.pending_final = None
                pattern.clear_pending_prediction()
            return False

        predicted = pending.get("side")
        correct = predicted == actual
        level_before = self.loss.level
        consec_before = self.loss.consec_losses
        # Loss gets stronger training weight. Example: L5/L6 losses cause more SGD passes.
        loss_boost = 1 if correct else min(10, max(3, consec_before + 3))
        learning_report = pattern.learn_from_actual(
            actual,
            actual_number=number,
            actual_color=color,
            period=period,
            loss_boost=loss_boost,
        )

        self.total_final += 1
        self.wins_final += int(correct)
        self.loss.on_result(correct)
        record_hourly_result(correct, self.loss.consec_losses)

        method = str(pending.get("source") or pending.get("method") or "AI_NN")
        if correct:
            telegram_recovery_alert(consec_before, level_before, period, method)
            reset_tg_level_alert()
        else:
            telegram_level_alert(self.loss.consec_losses, self.loss.level, predicted, actual, period, method)
            append_mistake(
                {
                    "period": period,
                    "prediction": predicted,
                    "actual": actual,
                    "actual_number": number,
                    "actual_color": color,
                    "method": method,
                    "level_before": level_before,
                    "consec_before": consec_before,
                    "paper_only": False,
                    "ai_learning": learning_report,
                    "ts_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        log.info(
            "FINAL settle %s -> %s | pred=%s actual=%s num=%s color=%s | level=%s consec=%s play=%s",
            period,
            "WIN" if correct else "LOSS",
            predicted,
            actual,
            number,
            color,
            self.loss.badge(),
            self.loss.consec_losses,
            "REAL",
        )
        self.pending_final = None
        return True

    def on_draw(self, period, number, color=None, publish_after=True):
        period = str(period).strip()
        number = int(number)
        color = pattern.clean_color(color, number)
        actual = number_to_size_letter(number)
        with self.lock:
            settled = self.settle(period, actual, number, color)
            # If there was no published pending prediction for this draw, still let
            # AI learn the completed result from the previous context.
            if not settled and len(self.history) >= pattern.MIN_HISTORY:
                try:
                    pattern.train_one(
                        list(self.history),
                        actual,
                        event_id=period,
                        numbers=list(self.history_nums),
                        colors=list(self.history_colors),
                        consec_losses=self.loss.consec_losses,
                        loss_boost=1,
                        actual_number=number,
                        actual_color=color,
                    )
                except Exception as exc:
                    log.warning("AI background train_one failed: %s", exc)
            self.history.append(actual)
            self.history_nums.append(number)
            self.history_colors.append(color)
            self.last_period = period
            self.draw_count += 1
        record_draw_row(period, number, color)
        if publish_after:
            self.publish(next_period(period))

    def publish(self, next_issue):
        with self.lock:
            self.loss.touch()
            hist = list(self.history)
            nums = list(self.history_nums)
            colors = list(self.history_colors)
            decision = pattern.predict(
                hist,
                event_id=next_issue,
                numbers=nums,
                colors=colors,
                consec_losses=self.loss.consec_losses,
                save=True,
            )
            side = decision.get("side") if decision.get("side") in ("B", "S") else "B"
            prediction = letter_to_label(side)
            confidence = int(decision.get("conf", pattern.CONF_FLOOR))
            meta = decision.get("meta") or {}
            method = meta.get("source") or "AI_NN"
            # Pause mode removed: play always stays active.
            bet_allowed = True
            play_signal = "PLAY"
            level_at_pred = self.loss.level
            self.pending_final = {
                "side": side,
                "period": next_issue,
                "conf": confidence,
                "method": method,
                "source": method,
                "ai_meta": meta,
                "level_at_pred": level_at_pred,
                "level_at_prediction": level_at_pred,
                "consec_losses_at_pred": self.loss.consec_losses,
                "bet_allowed": True,
                "play_signal": "PLAY",
                "paper_only": False,
            }
            win_rate = round(self.wins_final / self.total_final * 100, 1) if self.total_final else 0.0
            timestamp = ist_now().strftime("%Y-%m-%d %H:%M:%S")
            if self.loss.consec_losses == 0:
                level_desc = "NORMAL"
            elif self.loss.consec_losses >= 4:
                level_desc = f"AI_DEEP_LEARNING_RECOVERY(L{self.loss.level})"
            else:
                level_desc = "AI_RECOVER"
            self.latest = {
                "stream": STREAM_ID,
                "server": "DRAGO_AI_MAIN",
                "target": "size",
                "version": VERSION,
                "period": next_issue,
                "prediction": prediction,
                "prediction_code": side,
                "confidence": confidence,
                "level": level_at_pred,
                "level_at_pred": level_at_pred,
                "level_at_prediction": level_at_pred,
                "badge": self.loss.badge(),
                "consec_losses": self.loss.consec_losses,
                "level_desc": level_desc,
                "max_level_today": self.loss.max_level_today,
                "max_level_today_badge": self.loss.max_badge(),
                "max_consec_today": self.loss.max_consec_today,
                "stop_loss_hit": self.loss.stop_loss_hit,
                "stop_loss_streak": STOP_LOSS_STREAK,
                "paper_pause_removed": True,
                "bet_allowed": True,
                "play_signal": "PLAY",
                "paper_only": False,
                "risk_message": "OK to play",
                "level_cap": LEVEL_CAP,
                "method": method,
                "source": method,
                "status": "READY",
                "win_rate": win_rate,
                "wins": self.wins_final,
                "total": self.total_final,
                "draw_count": self.draw_count,
                "history_len": len(self.history),
                "clock_fallback_count": self.clock_fallback_count,
                "data_source": dict(DATA_SOURCE),
                "ai": meta,
                "learning_status": pattern.learning_status(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "timestamp": timestamp,
            }
            slim = {
                "server": "DRAGO_AI_MAIN",
                "version": VERSION,
                "period": next_issue,
                "prediction": prediction,
                "confidence": confidence,
                "level": level_at_pred,
                "level_at_pred": level_at_pred,
                "level_at_prediction": level_at_pred,
                "badge": self.loss.badge(),
                "consec_losses": self.loss.consec_losses,
                "level_desc": level_desc,
                "max_level_today": self.loss.max_level_today,
                "max_level_today_badge": self.loss.max_badge(),
                "max_consec_today": self.loss.max_consec_today,
                "stop_loss_hit": self.loss.stop_loss_hit,
                "paper_pause_removed": True,
                "bet_allowed": True,
                "play_signal": "PLAY",
                "paper_only": False,
                "risk_message": "OK to play",
                "source": method,
                "play_time": "N/A",
                "timestamp": timestamp,
            }
        self.last_publish_wall_ts = time.time()
        fb_put(FIREBASE_AUTOBET_URL, f"predictions/{STREAM_ID}", slim, FIREBASE_AUTOBET_SECRET)
        fb_put(FIREBASE_URL, f"predictions/{STREAM_ID}", self.latest, FIREBASE_SECRET)
        log.info(
            "PUBLISH %s -> %s conf=%s %s | level=%s consec=%s play=%s pB=%s",
            next_issue,
            prediction,
            confidence,
            method,
            self.loss.badge(),
            self.loss.consec_losses,
            self.latest.get("play_signal"),
            (self.latest.get("ai") or {}).get("prob_big"),
        )


ENGINE = Engine()


def clock_fallback_tick(reason="source_unavailable"):
    """Keep prediction endpoint moving if upstream history source is stale/down.

    This does not settle a real result (because no actual result was received),
    but it prevents the API/Firebase prediction from getting stuck after the
    external history API hits quota or temporarily stops updating.
    """
    if not CLOCK_FALLBACK_ENABLE or not ENGINE.bootstrapped:
        return False
    now_ts = time.time()
    with ENGINE.lock:
        if ENGINE.last_publish_wall_ts and now_ts - ENGINE.last_publish_wall_ts < CLOCK_FALLBACK_AFTER_SEC:
            return False
        pending_period = str((ENGINE.pending_final or {}).get("period") or "")
        base_period = pending_period or ENGINE.last_period
        if not base_period:
            return False
        # Pending prediction cannot be settled without actual result; void it and
        # advance one issue so clients do not see the same period forever.
        if ENGINE.pending_final:
            log.warning("clock fallback voided unsettled prediction %s (%s)", pending_period, reason)
            ENGINE.pending_final = None
            pattern.clear_pending_prediction()
        next_issue = next_period(base_period)
        ENGINE.clock_fallback_count += 1
        DATA_SOURCE["clock_fallback_active"] = True
        DATA_SOURCE["clock_fallback_count"] = ENGINE.clock_fallback_count
        DATA_SOURCE["clock_fallback_reason"] = str(reason)[:120]
    ENGINE.publish(next_issue)
    return True


def tick():
    if not ENGINE.bootstrapped:
        if not ENGINE.bootstrap_from_api():
            time.sleep(min(10, POLL_SEC))
        return
    records = fetch_history_api(POLL_HISTORY_LIMIT, force=False)
    # None means intentionally skipped due to throttle/backoff; do nothing.
    if records is None:
        return
    # [] means the source was contacted but returned no usable rows / failed.
    if not records:
        if CLOCK_FALLBACK_ENABLE:
            clock_fallback_tick(DATA_SOURCE.get("error") or "no_records")
        return
    with ENGINE.lock:
        last = ENGINE.last_period
    newer = [row for row in records if not last or period_sort_key(row["period"]) > period_sort_key(last)]
    if not newer:
        DATA_SOURCE["source_stale"] = True
        if CLOCK_FALLBACK_ENABLE:
            clock_fallback_tick("source_stale_no_new_period")
        return
    DATA_SOURCE["source_stale"] = False
    DATA_SOURCE["clock_fallback_active"] = False
    newer.sort(key=lambda row: period_sort_key(row["period"]))
    for row in newer[:-1]:
        ENGINE.on_draw(row["period"], row["number"], row.get("color"), publish_after=False)
    live = newer[-1]
    ENGINE.on_draw(live["period"], live["number"], live.get("color"), publish_after=True)


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
            requests.get(RENDER_KEEPALIVE_URL, timeout=15, headers={"User-Agent": "DRAGO-AI-KeepAlive/3.0"})
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


def build_daily_report(day=None):
    day = day or ist_today_str()
    with _hourly_lock:
        hours = dict(_hourly_stats.get(day) or {})
    if not hours:
        return f"📊 DRAGO AI DAILY REPORT\n📅 {day} IST\nNo settled predictions."
    now = ist_now()
    configured_end = max(0, min(24, SESSION_END_HOUR))
    completed_end = min(configured_end, now.hour)
    rows = []
    for hour in range(completed_end):
        bucket = hours.get(hour) or {"wins": 0, "losses": 0, "max_consec": 0}
        wins, losses = int(bucket["wins"]), int(bucket["losses"])
        total = wins + losses
        if total:
            rows.append((hour, wins, losses, total, wins / total, int(bucket["max_consec"])))
    if not rows:
        return f"📊 DRAGO AI DAILY REPORT\n📅 {day} IST\nNo completed-hour data yet."
    total_wins = sum(row[1] for row in rows)
    total_losses = sum(row[2] for row in rows)
    total = total_wins + total_losses
    overall_wr = total_wins / total if total else 0
    eligible = [row for row in rows if row[3] >= HOURLY_MIN_SAMPLE]
    best = max(eligible, key=lambda row: (row[4], row[3])) if eligible else None
    worst = min(eligible, key=lambda row: (row[4], -row[3])) if eligible else None
    max_cl = ENGINE.loss.max_consec_today
    lines = [
        "📊 DRAGO AI DAILY REPORT",
        "━━━━━━━━━━━━━━━━━━",
        f"📅 Date: {day} IST",
        f"🎮 Total Bets: {total}",
        f"✅ Wins: {total_wins}",
        f"❌ Losses: {total_losses}",
        f"🎯 Win Rate: {overall_wr * 100:.1f}%",
        f"🔥 Maximum Consecutive Loss: {max_cl}",
        f"⚠️ Risk Status: {risk_status(max_cl)}",
        f"📈 Highest Displayed Level: {ENGINE.loss.max_badge()}",
        f"🧠 AI: {pattern.learning_status().get('engine')}",
    ]
    if best:
        lines += ["", "🟢 BEST COMPLETED HOUR", f"{best[0]:02d}:00 | Bets {best[3]} | WR {best[4] * 100:.1f}% | CL {best[5]}"]
    if worst:
        lines += ["", "🔴 WEAKEST COMPLETED HOUR", f"{worst[0]:02d}:00 | Bets {worst[3]} | WR {worst[4] * 100:.1f}% | CL {worst[5]}"]
    return "\n".join(lines)


def daily_report_loop():
    last_sent = None
    while True:
        try:
            now = ist_now()
            key = f"{now.strftime('%Y-%m-%d')}-{DAILY_REPORT_HOUR}"
            if now.hour == DAILY_REPORT_HOUR and now.minute < 5 and last_sent != key:
                ok = telegram_send(build_daily_report(now.strftime("%Y-%m-%d")))
                if ok:
                    last_sent = key
            time.sleep(30)
        except Exception as exc:
            log.error("daily report loop: %s", exc)
            time.sleep(60)


@asynccontextmanager
async def lifespan(app):
    load_draw_store()
    ENGINE.loss.load_state()
    threads = (
        (poll_loop, "poll-30s-ai"),
        (render_keepalive_loop, "render-keepalive"),
        (midnight_cleanup_loop, "midnight"),
        (daily_report_loop, "daily-report"),
    )
    for target, name in threads:
        threading.Thread(target=target, daemon=True, name=name).start()
    telegram_send(
        "\n".join(
            [
                "🐉 DRAGO AI SERVER ONLINE",
                "━━━━━━━━━━━━━━━━━━",
                f"Version: {VERSION}",
                "Prediction: BIG/SMALL every valid round",
                "Old guards/rule prediction: REMOVED",
                "AI model: online neural network + loss learning",
                f"Bootstrap API limit: {BOOTSTRAP_HISTORY_LIMIT}",
                f"Level cap: L{LEVEL_CAP}" if LEVEL_CAP else "Level cap: UNLIMITED",
                f"Stop-loss signal: {STOP_LOSS_STREAK} losses" if STOP_LOSS_STREAK else "Stop-loss: disabled",
                "Ultra Safe paper-pause: REMOVED (play stays active)",
                f"Started: {ist_now().strftime('%d-%m-%Y %H:%M:%S')} IST",
            ]
        ),
        silent=True,
    )
    log.info("DRAGO AI started env=%s VPS_SECRET=%s", _ENV_FILE, "SET" if VPS_SECRET else "MISSING")
    yield
    pattern.save_state()


app = FastAPI(title="DRAGO AI WinGo 30s", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "app": "DRAGO AI Main",
        "version": VERSION,
        "stream": STREAM_ID,
        "mode": "Online neural AI; source fallback enabled; fake clock fallback hard-disabled; no paper-pause",
        "ultra_safe_mode": False,
        "paper_pause_removed": True,
        "ai": pattern.learning_status(),
    }


@app.get("/health")
def health():
    with ENGINE.lock:
        return {
            "ok": True,
            "version": VERSION,
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
            "ultra_safe_mode": False,
            "paper_pause_removed": True,
            "ultra_safe_pause_hit": False,
            "bet_allowed_now": True,
            "bootstrap_info": ENGINE.bootstrap_info,
            "clock_fallback_count": ENGINE.clock_fallback_count,
            "last_publish_age_sec": round(time.time() - ENGINE.last_publish_wall_ts, 1) if ENGINE.last_publish_wall_ts else None,
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
    payload = _draw_store_payload(items)
    payload["limit"] = limit
    return payload


@app.get("/wingo30s.json")
def wingo30s_json(limit: int = 100000):
    """Backward-compatible public JSON cache: data + savedAt, max 1 lakh rows."""
    limit = max(1, min(int(limit or 100000), DRAW_STORE_MAX))
    items = load_draw_store()[:limit]
    payload = _draw_store_payload(items)
    payload["limit"] = limit
    payload["file"] = DRAW_STORE_FILE
    return payload


@app.get("/api/debug")
def debug(request: Request):
    require_vps_auth(request)
    with ENGINE.lock:
        return {
            "latest": ENGINE.latest,
            "level": ENGINE.loss.level,
            "badge": ENGINE.loss.badge(),
            "consec_losses": ENGINE.loss.consec_losses,
            "last_history": list(ENGINE.history)[-60:],
            "last_numbers": list(ENGINE.history_nums)[-60:],
            "last_colors": list(ENGINE.history_colors)[-60:],
            "last_period": ENGINE.last_period,
            "learning_status": pattern.learning_status(),
            "learned_memory_top": pattern._state.top_patterns(20),
            "analysis_summary": {
                "count": ENGINE.analysis.get("count"),
                "side_pct": ENGINE.analysis.get("side_pct"),
                "number_pct": ENGINE.analysis.get("number_pct"),
                "color_pct": ENGINE.analysis.get("color_pct"),
                "streaks": ENGINE.analysis.get("streaks"),
            },
        }


@app.get("/api/ai/analysis")
def ai_analysis(request: Request):
    require_vps_auth(request)
    with ENGINE.lock:
        if ENGINE.analysis:
            return {"success": True, "analysis": ENGINE.analysis, "training": ENGINE.bootstrap_info}
    if os.path.exists(AI_ANALYSIS_FILE):
        try:
            with open(AI_ANALYSIS_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass
    items = load_draw_store()
    records = pattern.normalise_records(items)
    if records:
        analysis = pattern.deep_analyze_records(records)
        return {
            "success": True,
            "analysis": analysis,
            "training": ENGINE.bootstrap_info,
            "note": "generated from local draw store fallback",
        }
    return {"success": False, "detail": "analysis not ready", "data_source": dict(DATA_SOURCE)}


@app.get("/api/source/status")
def source_status(request: Request):
    require_vps_auth(request)
    return {
        "success": True,
        "data_source": dict(DATA_SOURCE),
        "bootstrapped": ENGINE.bootstrapped,
        "last_period": ENGINE.last_period,
        "clock_fallback_enabled": CLOCK_FALLBACK_ENABLE,
        "clock_fallback_count": ENGINE.clock_fallback_count,
        "real_source_required": not CLOCK_FALLBACK_ENABLE,
        "source_urls": [_safe_url(u) for u in _history_source_urls()],
        "draw_store_files": _draw_store_files(),
        "draw_store_count": len(load_draw_store()),
        "draw_store_max": DRAW_STORE_MAX,
        "history_api_min_interval_sec": HISTORY_API_MIN_INTERVAL_SEC,
        "history_api_backoff_sec": HISTORY_API_BACKOFF_SEC,
    }


@app.get("/api/source/resync")
def source_resync(request: Request, limit: int = 10000):
    require_vps_auth(request)
    records = fetch_history_api(max(1, min(int(limit or 10000), 10000)), force=True)
    if not records:
        return {"success": False, "detail": "source fetch failed", "data_source": dict(DATA_SOURCE)}
    ENGINE.load_records(records)
    ENGINE.analysis = pattern.deep_analyze_records(records)
    ENGINE.bootstrap_info = {"resync": True, "records": len(records), "source": DATA_SOURCE.get("source") or "history_api"}
    ENGINE.publish(next_period(ENGINE.last_period))
    return {"success": True, "records": len(records), "last_period": ENGINE.last_period, "latest": ENGINE.latest}


@app.get("/api/level/reset")
def level_reset(request: Request, clear_pending: int = 1, publish: int = 1):
    """Manual repair endpoint for polluted live state (for example old fake fallback L2)."""
    require_vps_auth(request)
    with ENGINE.lock:
        ENGINE.loss.consec_losses = 0
        ENGINE.loss.max_consec_today = 0
        ENGINE.loss.save_state()
        if int(clear_pending or 0):
            ENGINE.pending_final = None
            try:
                pattern.clear_pending_prediction()
            except Exception:
                pass
        last_period = ENGINE.last_period
    if int(publish or 0) and last_period:
        ENGINE.publish(next_period(last_period))
    return {
        "success": True,
        "level": ENGINE.loss.level,
        "consec_losses": ENGINE.loss.consec_losses,
        "cleared_pending": bool(int(clear_pending or 0)),
        "latest": ENGINE.latest,
    }


@app.get("/api/mistakes/recent")
def mistakes_recent(request: Request, limit: int = 20):
    require_vps_auth(request)
    limit = max(1, min(int(limit), 200))
    file_items = []
    if os.path.exists(MISTAKES_FILE):
        with _mistakes_lock:
            with open(MISTAKES_FILE, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        for line in lines[-limit:]:
            try:
                file_items.append(json.loads(line))
            except Exception:
                pass
    file_items.reverse()
    return {"count": len(file_items), "items": file_items, "ai_recent_mistakes": pattern.recent_mistakes(limit)}


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
        "ai": pattern.learning_status(),
    }


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


@app.get("/api/patterns/top")
def top_patterns(request: Request, limit: int = 20):
    require_vps_auth(request)
    limit = max(1, min(int(limit), 100))
    return {
        "note": "Old fixed patterns/guards removed. These are learned AI loss-memory contexts.",
        "total_contexts": pattern._state.total_patterns(),
        "reliable_contexts": pattern._state.reliable_patterns(),
        "top": pattern._state.top_patterns(limit),
    }


if __name__ == "__main__":
    log.info("starting DRAGO AI on port %s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
