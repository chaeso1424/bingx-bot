# app.py
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

from utils.logging import log
from models.config import BotConfig
from models.state import BotState
from services.bingx_client import BingXClient, BASE, _req_get, _ts
from bot.runner import BotRunner

LOG_FILE = Path("./logs.txt")

# ───────────────────────────────────────────────────────────────────────────────
# 0) 환경 로드 (.env는 파일 위치 기준으로 확실히 읽기)
# ───────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # 명시 경로
load_dotenv()                   # (추가) 현재 작업 디렉토리 기준

# ───────────────────────────────────────────────────────────────────────────────
# 1) 설정 파일 경로 / ENV 플래그
# ───────────────────────────────────────────────────────────────────────────────
CONFIG_FILE = Path("./data/config.json")
APPLY_ON_SAVE = os.getenv("APPLY_ON_SAVE", "true").lower() == "true"
SKIP_SETUP    = os.getenv("SKIP_SETUP", "false").lower() == "true"

# ───────────────────────────────────────────────────────────────────────────────
# 2) Flask 앱/전역 상태
# ───────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev")

state = BotState()
cfg = BotConfig(
    symbol=os.getenv("DEFAULT_SYMBOL", "BTC-USDT"),
    side=os.getenv("DEFAULT_SIDE", "BUY"),
)

client = BingXClient()
runner = BotRunner(cfg, state, client)

# ───────────────────────────────────────────────────────────────────────────────
# 3) 설정 저장/로드 헬퍼
# ───────────────────────────────────────────────────────────────────────────────
def persist_cfg(cfg_obj: BotConfig):
    """현재 cfg를 로컬 파일에 저장"""
    doc = {
        "symbol": cfg_obj.symbol,
        "side": cfg_obj.side,
        "margin_mode": cfg_obj.margin_mode,
        "leverage": int(cfg_obj.leverage),
        "tp_percent": float(cfg_obj.tp_percent),
        "dca_config": cfg_obj.dca_config,  # [[gap, usdt], ...]
        "repeat_mode": bool(getattr(state, "repeat_mode", False)),
    }
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

def load_cfg_into(cfg_obj: BotConfig):
    """저장된 설정을 읽어 cfg 객체에 주입 (없으면 무시)"""
    if not CONFIG_FILE.exists():
        return
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        doc = json.load(f)

    cfg_obj.symbol = doc.get("symbol", getattr(cfg_obj, "symbol", "BTC-USDT"))
    cfg_obj.side = doc.get("side", getattr(cfg_obj, "side", "BUY"))
    cfg_obj.margin_mode = doc.get("margin_mode", getattr(cfg_obj, "margin_mode", "CROSS"))
    try:
        cfg_obj.leverage = int(doc.get("leverage", getattr(cfg_obj, "leverage", 10)))
    except Exception:
        pass
    try:
        cfg_obj.tp_percent = float(doc.get("tp_percent", getattr(cfg_obj, "tp_percent", 4.5)))
    except Exception:
        pass
    dca = doc.get("dca_config")
    if isinstance(dca, list) and dca:
        cfg_obj.dca_config = dca
    # 반복 모드 복원(선택)
    try:
        state.repeat_mode = bool(doc.get("repeat_mode", False))
    except Exception:
        state.repeat_mode = False

# 초기 로드
load_cfg_into(cfg)

# ───────────────────────────────────────────────────────────────────────────────
# 4) 라우트
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    try:
        symbols = client.list_symbols()
    except Exception as e:
        log(f"⚠️ 심볼 조회 실패: {e}")
        symbols = [cfg.symbol]
    try:
        avail = client.get_available_usdt()
    except Exception:
        avail = 0.0

    return render_template(
        "index.html",
        running=state.running,
        repeat_mode=getattr(state, "repeat_mode", False),
        cfg=cfg,
        symbols=symbols,
        available_usdt=avail,
    )

@app.route("/logs")
def logs_page():
    return render_template("logs.html")


@app.route("/logs/text")
def logs_text():
    """마지막 500줄 로그를 내려주는 단순 엔드포인트"""
    tail = 500
    try:
        LOG_FILE.touch(exist_ok=True)
        with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        body = "".join(lines[-tail:])
    except Exception:
        body = ""
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return Response(body, mimetype="text/plain", headers=headers)

@app.get("/symbols")
def symbols():
    try:
        return jsonify(client.list_symbols())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/status")
def status():
    try:
        pp, qp = client.get_symbol_filters(cfg.symbol)
    except Exception:
        pp, qp = 4, 0
    try:
        mark = float(client.get_mark_price(cfg.symbol))
    except Exception:
        mark = None

    avg, qty = client.position_info(cfg.symbol, cfg.side)
    exch_lev = client.get_current_leverage(cfg.symbol, cfg.side)

    out = {
        "running": state.running,
        "repeat_mode": getattr(state, "repeat_mode", False),
        "tp_order_id": getattr(state, "tp_order_id", None),
        "symbol": cfg.symbol,
        "side": cfg.side,
        "price_precision": pp,
        "position_avg_price": avg,
        "position_qty": qty,
        "mark_price": mark,
        "exchange_leverage": exch_lev,
        "cfg_leverage": cfg.leverage,
        "budget_required": getattr(state, "budget_required", None),
        "budget_available": getattr(state, "budget_available", None),
        "budget_ok": getattr(state, "budget_ok", None),
    }
    return jsonify(out)

@app.post("/config")
def save_config():
    data = request.get_json(force=True)

    cfg.symbol = data["symbol"]
    cfg.side = data["side"]
    cfg.margin_mode = data["margin_mode"]  # "CROSS" | "ISOLATED"
    cfg.leverage = int(data["leverage"])
    cfg.tp_percent = float(data["tp_percent"])
    cfg.dca_config = data["dca_config"]

    # 파일로 저장
    persist_cfg(cfg)

    # 저장 즉시 거래소 반영 (환경변수로 제어)
    if APPLY_ON_SAVE and not SKIP_SETUP:
        # 마진 모드
        try:
            client.set_margin_mode(cfg.symbol, cfg.margin_mode)
        except Exception as e:
            # 저장은 성공시키고 경고만 반환(200)
            return jsonify({
                "ok": False,
                "cfg": {
                    "symbol": cfg.symbol, "side": cfg.side,
                    "margin_mode": cfg.margin_mode, "leverage": cfg.leverage,
                    "tp_percent": cfg.tp_percent, "dca_config": cfg.dca_config
                },
                "error": f"margin_mode 적용 실패: {e}"
            }), 200

        # 레버리지
        try:
            client.set_leverage(cfg.symbol, cfg.leverage)
        except Exception as e:
            return jsonify({
                "ok": False,
                "cfg": {
                    "symbol": cfg.symbol, "side": cfg.side,
                    "margin_mode": cfg.margin_mode, "leverage": cfg.leverage,
                    "tp_percent": cfg.tp_percent, "dca_config": cfg.dca_config
                },
                "error": f"leverage 적용 실패: {e}"
            }), 200

    return jsonify({
        "ok": True,
        "cfg": {
            "symbol": cfg.symbol, "side": cfg.side,
            "margin_mode": cfg.margin_mode, "leverage": cfg.leverage,
            "tp_percent": cfg.tp_percent, "dca_config": cfg.dca_config
        }
    })

@app.post("/repeat")
def toggle_repeat():
    try:
        state.repeat_mode = not getattr(state, "repeat_mode", False)
        # (선택) 반복 모드 저장 유지
        try:
            persist_cfg(cfg)
        except Exception:
            pass
        return jsonify({"ok": True, "repeat": state.repeat_mode})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/start")
def start():
    try:
        avail = client.get_available_usdt()
    except Exception as e:
        return jsonify({"ok": False, "msg": f"잔고 확인 실패: {e}"}), 400

    if avail <= 0:
        return jsonify({"ok": False, "msg": "가용 USDT가 0입니다. 금액/레버리지/마진모드 확인"}), 400

    if not state.running:
        runner.start()
        return jsonify({"ok": True})

    return jsonify({"ok": False, "msg": "이미 실행 중"}), 400

@app.post("/stop")
def stop():
    runner.stop()
    return jsonify({"ok": True})

@app.get("/debug/balance")
def debug_balance():
    try:
        val = client.get_available_usdt()
        return jsonify({"ok": True, "available_usdt": val})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/debug/balance/raw")
def debug_balance_raw():
    """
    서명 GET으로 /user/balance 원본문서 그대로 조회
    """
    try:
        url = f"{BASE}/openApi/swap/v2/user/balance"
        j = _req_get(url, {"recvWindow": 60000, "timestamp": _ts()}, signed=True)
        return jsonify({"ok": True, "json": j})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ───────────────────────────────────────────────────────────────────────────────
# 5) 엔트리포인트
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 개발 편의를 위한 옵션(필요시 수정)
    app.run(debug=True)

