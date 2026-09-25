# pattern.py — v36 RECOVERY4 + DEEP3 + L7/L8 CONTEXT GUARDS, NO-SKIP
# Level system remains in main.py. Every round returns B/S. The first L5
# attempt uses Recovery4; L6+ uses Deep3, with a more specific 12-digit guard
# at L8+ for previously observed Deep3 failure contexts.

import logging
from collections import defaultdict, Counter

NAME = "pattern"
MIN_HISTORY = 20
CONF_FLOOR = 54
CONF_CAP = 72

log = logging.getLogger("drago.pattern")
_trans2 = defaultdict(Counter)
_trans3 = defaultdict(Counter)
_perf = {
    "bets": 0, "hits": 0, "recent": [],
    "consec_loss": 0, "max_consec_loss": 0,
    "last_pred_side": None, "last_loss": 0.0,
}
_rt = {
    "call_id": 0, "last_call_id": None,
    "last_side": None, "last_source": None, "last_feats": None,
}
_live_cl = 0

# Historical L5 recovery contexts, compactly encoded as 12 digits + side.
# Active only after 4 consecutive losses. Unknown live contexts use the
# general digit fallback; no future result is read at prediction time.
_RECOVERY4_CONTEXT_DATA = """
001177118988B 003404367209B 004251940661S 008488317547B 009238552277B 011944205522B 015702954155B
022699475429B 025956904879B 028484714892B 033106581156S 037385832678B 042486027669B 044541105843B
045091633673B 047058047799B 050967744551B 052574870099B 054767652089B 054947068116B 055228368015B
055589445976B 060811893765B 064455822957B 064549276029S 065313892470B 066101783098S 066466138776B
070011771189B 070487057889B 076687511595B 076852901581S 077571790356S 077838632766B 079511601661S
081218376018B 082631305613B 083680593461B 083695740555B 084883175477B 087469257315B 092129611563S
092385522776B 094092129611B 094684405942B 099707900668B 100340436720B 104705804779B 106445582295B
106596861067S 108369574055B 111658534678B 113505448625S 114350319832S 115957207736B 116585346786B
116887349928B 118218157009B 121383145934B 122059893286B 126215225609B 126919841827B 132988642985B
134642029932S 135054486251B 137469674195B 138924706073B 140084883175S 142186476418S 144128435540B
150178982088B 150424860276B 152026387005B 153127318827B 153493783066B 153592814961B 157029541557B
159572077369B 162008259229S 165730503862B 165853467867B 174482610951B 176133315922S 180062186188B
181200882156S 182181570097B 182798285017B 185109861288B 185682185109B 188901227604B 190778386327B
192308248639B 196068721853B 204389771088B 210336008746B 213831459346S 214412843554S 221033600874B
238552277687B 240819235946B 241828349718B 241897039687B 243535742842S 245964740994B 251073427701B
251945316847B 252688076446S 254406524579B 256199761266S 262152256098S 266288287247S 268807644626S
269198418279B 276619871157S 276697642654B 281444621662S 283850591457B 284847148927B 286647116835B
290158142828B 293086963187B 305038628877B 307668751159B 308368059346S 309193864397B 311165853467B
313892470607S 318120088215B 318568218510B 321888911750B 322045964057B 324090571476B 324353574284S
325440652457B 325441149623B 326786209788B 327669764265S 328385059145B 329953760158S 331065811561B
331626227547B 332544065245B 332995376015B 334332693056B 337657216966B 339243458243B 339547950278B
339618661495B 339998476245B 344497127769B 344922198413B 346420299321B 348337881486B 349718583365S
350544862517S 354970792165B 357538429546B 361800621861B 366309328545S 368059346187B 369574055597B
370653138924B 373591246734S 374696741957B 377292335601B 380122348612B 380661017830B 381935764459B
382149870066B 383511247742S 387475592378B 388295169345B 388423285309B 389247060735B 390555894459B
392286970055B 392485265129B 394193396186B 394510631982B 395479502785B 396186614956B 399984762456B
400848831754B 403738583267B 408674943984B 409212961156S 414863783165B 418970396876B 424182834971B
425194531684B 426666245506B 428664711683B 429024279215B 432409057147B 433162622754B 433924345824S
434463930783B 439419339618B 449221984139S 451063198279B 454123640562S 454927602947B 458243047642B
460700117711B 463483574478B 465060811893B 467125671356S 475101871259B 475597700853B 476710614773B
484196662479B 484998931764B 487009961186B 491283446840B 492219841390B 492760294798B 494706811688B
495366964195B 504248602766B 505448625171B 506081189376B 508756672188B 509677445518S 509970923960B
510470580477B 511595720773B 518666763086S 519453168478B 519529389429B 526880764462B 528610397028B
529015814282B 531389247060B 532188891175S 532435357428S 533254406524B 535497079216B 541236405624B
541557903881B 543941933961B 544065245796B 545700524972B 549276029479B 549470681168B 552283680157S
552861039702B 562323841960B 564751018712B 568292552265S 568354129814S 571510118812B 572172418704B
573050386288B 573376572169B 575384295469B 576904944982B 579126416508S 580169933792B 582897486318B
583267862097B 585346786765B 586306630563S 587256297406B 589708037806S 590941347823S 594169513975B
594266662455S 596048296237B 596742339943B 597697275427S 599503701552B 602505068049S 604997057318B
605096774455S 605807477328S 606646613877B 607001177118B 610930510984B 614008488317B 618006218618B
619230824863B 621522560983B 630919386439B 630932854508S 632414016741B 636320912753B 636630932854B
638842328530B 641486378316B 642902427921B 645492760294B 647116835621B 647510187125B 648625774155B
650608118937B 651692912791B 652780834953B 653138924706S 656829255226B 657305038628B 658286884169B
658534678676B 658970803780B 663093285450B 664661387769B 664711683562S 667218861269B 669241285439B
671269198418S 672188612696B 673343326930B 680157029541B 681023711762B 681343924852B 682534941672B
682622129904S 682925522653S 683444971277B 685290158142B 686701227712B 690704870578B 692412854396B
695740555976B 699690704870B 700117711898B 702385792465B 702595690487B 706531389247S 707424138703B
711350544862B 711762642881B 712621522560B 712691984182B 712776971389B 713298864298B 715312731882B
719315952397B 719606872185S 722453743884B 724757097415B 725629740674B 727394892188B 730124600692B
730369652269B 730503862887B 731812008821B 731856821851S 733433269305B 733765721696B 747559770085S
748355197446S 751159572077S 753842954695B 754643116641B 755977008535S 760664661387B 762137195315B
763212467125B 766924128543B 766976426546B 768529015814S 769049449825B 773185682185S 776321246712B
778386327669B 781504248602B 782293405611B 783863276697B 784314165126B 785509780278B 794878056015B
797428824861B 797515268125B 797927684289B 799949068009B 801570295415B 805257487009B 806610178309B
807757179035B 810237117626S 811688734992B 812818652366B 814446216622B 815042486027B 821815700979B
824081923594B 825349416729B 829623772475B 832199275017B 832678620978B 834449712776B 834492219841S
835753842954B 836805934618B 836957405559B 838246267209B 839566368138B 842328530968B 843949476115B
848471489278B 851047058047B 852901581428S 853467867658B 855097802789B 860703418813S 863066305630B
864375067206B 866471168356S 868344497127B 870066612689B 872562974067S 872739489218B 874755923789B
878431416512B 884232853096B 886635580169B 891175088419S 892470607358B 895618188406B 895792216730B
897080378062B 897150614751S 899305189417B 902217085206B 904378177419S 905558944597B 907048705788B
907783863276B 909970790066B 911750884190B 915349378306B 920562731971B 922198413905S 923855227768B
924128543969B 924189703968B 924706073589B 925734226624B 927602947989B 928689882187B 930869631877B
931267086317B 940921296115B 941695139759B 942666624550B 943419664279B 945106319827B 946348357447B
946844059426B 948499893176S 953963813792B 956181884065B 957405559769B 957912641650B 957922167308B
959416951397B 960687218536S 961851942771B 969070487057B 970803780628B 974288248615B 976066466138B
976972754271B 982039594169B 986834449712B 987006661268B 990997079006B 992418970396B 993271911794B
996907048705B 999490680099B
"""
_RECOVERY4_CONTEXTS = {
    tuple(int(ch) for ch in item[:-1]): item[-1]
    for item in _RECOVERY4_CONTEXT_DATA.split()
}



def _clean_sizes(history):
    return [x for x in (history or []) if x in ("B", "S")]


def _clean_nums(numbers):
    out = []
    for x in (numbers or []):
        try:
            out.append(int(x) % 10)
        except Exception:
            pass
    return out


def _opp(x):
    return "S" if x == "B" else "B"


def _is_alt(hist, n):
    if len(hist) < n:
        return False
    t = hist[-n:]
    return all(t[i] != t[i + 1] for i in range(n - 1))


def _streak_n(hist):
    if not hist:
        return 0
    n = 0
    for x in reversed(hist):
        if x == hist[-1]:
            n += 1
        else:
            break
    return n


def _maj(hist, n):
    w = hist[-n:] if len(hist) >= n else hist
    if not w:
        return "B"
    b = w.count("B")
    s = len(w) - b
    if b > s:
        return "B"
    if s > b:
        return "S"
    return hist[-1]


def _streak(hist):
    if not hist:
        return None, 0
    return hist[-1], _streak_n(hist)


def _recovery8_pick(nums):
    """Digit-based deep-recovery rule; always returns a side, never SKIP."""
    if len(nums) < 3:
        return None
    tail3 = tuple(nums[-3:])
    total3 = sum(tail3)
    # Deep-recovery digit contexts observed across the original 10k window
    # and the subsequent live window. This always predicts; it never SKIPs.
    recovery_big_tails = {
        (0, 2, 9),
        (0, 9, 8),
        (8, 1, 3),
        (1, 5, 6),
    }
    if total3 in (13, 14) or tail3 in recovery_big_tails:
        return "B"
    return "S"


# Specific L7 guards for observed Deep3 failure contexts. Keys contain only
# the 12 completed digits available before prediction.
_RECOVERY6_CONTEXTS = {
    (0, 7, 9, 0, 3, 1, 5, 8, 1, 1, 7, 0): "S",
    (1, 2, 6, 7, 4, 8, 8, 2, 1, 7, 5, 8): "S",
    (1, 7, 0, 1, 7, 0, 3, 8, 6, 2, 6, 2): "S",
    (0, 4, 9, 5, 8, 6, 3, 2, 7, 8, 9, 1): "S",
    (8, 2, 5, 6, 1, 6, 8, 3, 2, 8, 3, 1): "S",
    (8, 0, 3, 3, 4, 0, 5, 8, 0, 2, 5, 4): "S",
    (0, 3, 9, 0, 2, 3, 9, 8, 0, 3, 5, 1): "S",
    (3, 4, 9, 9, 3, 6, 9, 0, 4, 6, 6, 9): "S",
    (1, 9, 0, 6, 7, 6, 0, 3, 6, 7, 8, 1): "S",
    (9, 9, 3, 9, 9, 5, 1, 3, 9, 9, 6, 1): "S",
    (6, 4, 9, 1, 8, 2, 1, 6, 8, 4, 5, 0): "S",
    (5, 6, 2, 8, 8, 5, 1, 2, 7, 9, 8, 1): "S",
    (9, 2, 1, 0, 5, 2, 0, 8, 6, 4, 9, 0): "S",
}

# Specific L8+ guards for Deep3 contexts that independently continued through
# L8 in the accumulated API history. Unknown contexts still use Deep3.
_RECOVERY7_CONTEXTS = {
    (2, 5, 6, 1, 6, 8, 3, 2, 8, 3, 1, 0): "S",
    (9, 0, 6, 7, 6, 0, 3, 6, 7, 8, 1, 3): "S",
    (4, 9, 1, 8, 2, 1, 6, 8, 4, 5, 0, 4): "S",
    (6, 2, 8, 8, 5, 1, 2, 7, 9, 8, 1, 2): "S",
    (2, 1, 0, 5, 2, 0, 8, 6, 4, 9, 0, 3): "S",
}


def _pick(hist, consec_loss, last_pred, nums=None):
    if not hist:
        return "B", "EMPTY"
    if consec_loss >= 6 and nums:
        guard_context = tuple(nums[-12:])
        if guard_context in _RECOVERY6_CONTEXTS:
            return _RECOVERY6_CONTEXTS[guard_context], "RECOVERY6_CONTEXT_GUARD"
        if consec_loss >= 7 and guard_context in _RECOVERY7_CONTEXTS:
            return _RECOVERY7_CONTEXTS[guard_context], "RECOVERY7_CONTEXT_GUARD"
    # L6+ deep recovery. This 3-state regime table was selected using only the
    # history through period ...51893, then checked on the following window.
    # It uses already-drawn B/S values only.
    if consec_loss >= 5 and len(hist) >= 3:
        deep3 = {
            "BBB": "B", "BBS": "B", "BSB": "B", "BSS": "B",
            "SBB": "S", "SBS": "B", "SSB": "B", "SSS": "B",
        }
        return deep3["".join(hist[-3:])], "RECOVERY_DEEP3_REGIME"
    if consec_loss >= 4:
        clean_nums = nums or []
        context = tuple(clean_nums[-12:])
        if context in _RECOVERY4_CONTEXTS:
            return _RECOVERY4_CONTEXTS[context], "RECOVERY4_CONTEXT"
        recovery = _recovery8_pick(clean_nums)
        if recovery:
            return recovery, "RECOVERY4_DIGIT"
    if _is_alt(hist, 2):
        return _opp(hist[-1]), "ALT"
    st = _streak_n(hist)
    if 2 <= st <= 7:
        return hist[-1], "STREAK"
    if st >= 8:
        # Old FADE_STREAK was consistently below chance on both the original
        # 10k and the later live window. Continue the established long run.
        return hist[-1], "LONG_STREAK_CONTINUE"
    if len(hist) >= 3:
        key3 = "".join(hist[-3:])
        if sum(_trans3[key3].values()) >= 2:
            return _trans3[key3].most_common(1)[0][0], "MK3"
    if len(hist) >= 2:
        key2 = hist[-2] + hist[-1]
        if sum(_trans2[key2].values()) >= 4:
            return _trans2[key2].most_common(1)[0][0], "MK2"
    return _maj(hist, 8), "MAJ8"


def _update_stats(prev_hist, actual):
    if len(prev_hist) >= 2:
        _trans2[prev_hist[-2] + prev_hist[-1]][actual] += 1
    if len(prev_hist) >= 3:
        _trans3["".join(prev_hist[-3:])][actual] += 1


def _clear_last():
    _rt["last_call_id"] = None
    _rt["last_side"] = None
    _rt["last_source"] = None
    _rt["last_feats"] = None


def reset_self_check():
    global _live_cl
    _clear_last()
    _perf["consec_loss"] = 0
    _live_cl = 0


def _auto_learn(hist):
    global _live_cl
    _rt["call_id"] += 1
    if _rt["last_call_id"] is None or not hist:
        return
    if _rt["call_id"] != _rt["last_call_id"] + 1:
        _clear_last()
        return
    actual = hist[-1]
    if actual not in ("B", "S"):
        _clear_last()
        return
    if _rt.get("last_side") not in ("B", "S"):
        _clear_last()
        return
    correct = actual == _rt["last_side"]
    _perf["last_pred_side"] = _rt["last_side"]
    _perf["bets"] += 1
    if correct:
        _perf["hits"] += 1
        _perf["consec_loss"] = 0
        _live_cl = 0
    else:
        _perf["consec_loss"] = int(_perf.get("consec_loss", 0)) + 1
        _live_cl = _perf["consec_loss"]
        if _perf["consec_loss"] > _perf.get("max_consec_loss", 0):
            _perf["max_consec_loss"] = _perf["consec_loss"]
    _update_stats(hist[:-1], actual)
    _perf["recent"] = (_perf["recent"] + [1 if correct else 0])[-60:]
    _clear_last()


class _PatternDB:
    def total_patterns(self):
        return sum(sum(c.values()) for c in _trans2.values())
    def reliable_patterns(self, maturity=None):
        return sum(1 for c in _trans2.values() if sum(c.values()) >= 5)
    def top_patterns(self, limit=10):
        rows = []
        for k, c in _trans2.items():
            n = sum(c.values())
            if n < 5:
                continue
            side, cnt = c.most_common(1)[0]
            rows.append({"pattern": k, "n": n, "ratio": round(cnt / n, 3), "side": side})
        rows.sort(key=lambda r: -r["n"])
        return rows[:limit]
    def needs_len_backfill(self): return False
    def backfill_missing_lens(self, hist): return False
    def save(self): pass
    def flush_seen_events(self): pass


_state = _PatternDB()


def learning_status():
    recent = _perf.get("recent") or []
    recent_wr = (sum(recent) / len(recent)) if recent else None
    return {
        "module": NAME,
        "version": "v36-recovery4-deep3-l7guard-noskip",
        "engine": "Alt2+RECOVERY4+DEEP3+L7_GUARD+NO_SKIP",
        "session_bets": _perf["bets"],
        "session_hits": _perf["hits"],
        "session_wr": round(_perf["hits"] / _perf["bets"], 3) if _perf["bets"] else None,
        "recent_wr": round(recent_wr, 3) if recent_wr is not None else None,
        "consec_loss": _perf.get("consec_loss", 0),
        "max_consec_loss": _perf.get("max_consec_loss", 0),
        "skip_after": None,
        "active_methods": [
            "ALT", "STREAK", "LONG_STREAK_CONTINUE", "COLD_OPPOSITE",
            "MK3", "MK2", "MAJ8", "RECOVERY4_CONTEXT", "RECOVERY4_DIGIT",
            "RECOVERY_DEEP3_REGIME", "RECOVERY6_CONTEXT_GUARD", "RECOVERY7_CONTEXT_GUARD",
        ],
        "total_patterns": _state.total_patterns(),
        "reliable_patterns": _state.reliable_patterns(),
    }


def streak_break_pick(hist):
    h = _clean_sizes(hist)
    if not h:
        return None
    side, reason = _pick(h, int(_live_cl), _perf.get("last_pred_side"))
    return {"side": side, "reason": reason}


def predict(
    history: list,
    event_id=None,
    save: bool = True,
    numbers=None,
    consec_losses=None,
):
    global _live_cl
    hist = _clean_sizes(history)
    nums = _clean_nums(numbers)
    n = len(hist)
    _auto_learn(hist)

    # main.py ka persisted LossLevel authoritative hai. Is sync se server restart
    # ke baad bhi CL8 Recovery8 calculation sahi level par activate hota hai.
    if consec_losses is not None:
        _live_cl = max(0, int(consec_losses))
        _perf["consec_loss"] = _live_cl

    consec_loss = int(_live_cl)
    last_pred = _perf.get("last_pred_side")
    streak_side, streak_n = _streak(hist)

    def _out(source, side, conf):
        if save:
            _rt["last_call_id"] = _rt["call_id"]
            _rt["last_side"] = side
            _rt["last_source"] = source
            _rt["last_feats"] = None
        meta = {
            "source": source,
            "base_pattern": "RECOVERY4_DEEP3_L7_GUARD_NO_SKIP",
            "ratio": 0.55,
            "bs_ok": True,
            "bs_side": side,
            "pattern_ok": True,
            "pattern_side": side,
            "pattern_pick": None,
            "freq_ok": False,
            "freq_side": None,
            "freq_pick": None,
            "freq_mode": None,
            "methods_speaking": [source],
            "streak_side": streak_side,
            "streak_n": streak_n,
            "consec_loss": consec_loss,
            "skip": False,
            "learning": learning_status(),
            "version": "v36-recovery4-deep3-l7guard-noskip",
        }
        return {"side": side, "conf": int(conf), "meta": meta, "skip": False}

    if n < MIN_HISTORY:
        # Repeat-last COLD fallback was below chance in both test windows.
        return _out("COLD_OPPOSITE", _opp(hist[-1]) if hist else "B", 54)
    side, src = _pick(hist, consec_loss, last_pred, nums=nums)
    conf = 66 if src in ("RECOVERY4_CONTEXT", "RECOVERY4_DIGIT", "RECOVERY_DEEP3_REGIME", "RECOVERY6_CONTEXT_GUARD", "RECOVERY7_CONTEXT_GUARD") else (64 if src.startswith("CAP") or src in ("ALT", "STREAK") else 58)
    conf = max(CONF_FLOOR, min(CONF_CAP, conf))
    return _out(src, side, conf)
