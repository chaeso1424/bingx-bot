# services/bingx_client.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from typing import Any

from utils.logging import log

# --- force IPv4 only for urllib3/requests (환경에 따라 DNS/IPv6 이슈 회피) ---
import socket
try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

SKIP_SETUP = os.getenv("SKIP_SETUP", "false").lower() == "true"

API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
BASE = os.getenv("BINGX_BASE", "https://open-api.bingx.com").rstrip("/")
POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "HEDGE").upper()  # HEDGE or ONEWAY

TIMEOUT = (5, 60)  # (connect, read)

# ---------- low-level utils ----------
def _ts() -> int:
    return int(time.time() * 1000)


def _headers(form: bool = False):
    if not API_KEY:
        raise RuntimeError("BINGX_API_KEY is empty. Check your .env and app load order.")
    h = {"X-BX-APIKEY": API_KEY}
    h["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
    h["User-Agent"] = "bingx-bot/1.0"
    return h


def _sign(params: dict) -> str:
    """Return "query_string&signature=..." for signed requests."""
    qs = urlencode(params)
    sig = hmac.new(API_SECRET.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig


# --- normalize payload for signed requests (bool -> "true"/"false") ---
def _coerce_params(d: dict | None) -> dict:
    out: dict[str, str] = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        elif v is None:
            # 서명 파라미터에 None은 넣지 않음
            continue
        else:
            out[k] = str(v)
    return out


def _inject_signed_defaults(params: dict | None) -> dict:
    """모든 서명 요청에 recvWindow/timestamp 기본값을 보장."""
    p = dict(params or {})
    p.setdefault("recvWindow", 60000)
    p.setdefault("timestamp", _ts())
    return p


def _req_get(url: str, params: dict | None = None, signed: bool = False) -> dict:
    params = params or {}
    if signed:
        params = _inject_signed_defaults(params)
        params = _coerce_params(params)
        qs = _sign(params)
        r = requests.get(url + "?" + qs, headers=_headers(), timeout=TIMEOUT)
    else:
        r = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT)
    try:
        j = r.json()
    except Exception:
        r.raise_for_status()
        raise
    code = str(j.get("code", "0"))
    if code != "0":
        msg = j.get("msg") or j
        raise RuntimeError(f"BingX error @GET {url}: {code} {msg}")
    return j


def _req_delete(url: str, params: dict | None = None, signed: bool = False) -> dict:
    params = params or {}
    if signed:
        params = _inject_signed_defaults(params)
        params = _coerce_params(params)
        qs = _sign(params)
        r = requests.delete(url + "?" + qs, headers=_headers(form=True), timeout=TIMEOUT)
    else:
        r = requests.delete(url, params=params, headers=_headers(form=False), timeout=TIMEOUT)
    j = r.json()
    code = str(j.get("code", "0"))
    if code != "0":
        raise RuntimeError(f"BingX error @DELETE {url}: {code} {j.get('msg')}")
    return j


def _req_post(url: str, body: dict | None = None, signed: bool = False) -> dict:
    body = body or {}
    if signed:
        body = _inject_signed_defaults(body)
        body = _coerce_params(body)
        payload = _sign(body)  # querystring + signature
        r = requests.post(url, data=payload, headers=_headers(form=True), timeout=TIMEOUT)
    else:
        r = requests.post(url, json=body, headers=_headers(form=False), timeout=TIMEOUT)
    j = r.json()
    code = str(j.get("code", "0"))
    if code != "0":
        raise RuntimeError(f"BingX error @POST {url}: {code} {j.get('msg')}")
    return j


# ---------- high-level client ----------
class BingXClient:
    def __init__(self):
        self._spec_cache: dict[str, dict] = {}

    # (없으면 추가) 주문 응답에서 orderId 추출
    def _extract_order_id(self, resp: dict) -> str:
        """
        BingX 주문 응답에서 orderId를 최대한 유연하게 추출한다.
        지원 케이스:
        - resp["orderId"] / resp["id"]
        - resp["data"]["orderId"] / resp["data"]["id"]
        - resp["data"]["order"]["orderId"/"orderID"/"id"]
        """
        if not isinstance(resp, dict):
            return ""
        for k in ("orderId", "orderID", "id"):
            v = resp.get(k)
            if v:
                return str(v)
        d = resp.get("data")
        if isinstance(d, dict):
            for k in ("orderId", "orderID", "id"):
                v = d.get(k)
                if v:
                    return str(v)
            o = d.get("order")
            if isinstance(o, dict):
                for k in ("orderId", "orderID", "id", "order_id"):
                    v = o.get(k)
                    if v:
                        return str(v)
        return ""

    def get_symbol_filters(self, symbol: str) -> tuple[int, int]:
        """
        /quote/contracts에서 symbol의 pricePrecision, quantityPrecision을 얻어온다.
        실패 시 (4, 3) 반환.
        """
        try:
            url = f"{BASE}/openApi/swap/v2/quote/contracts"
            j = _req_get(url)
            data = j.get("data", [])
            for it in data if isinstance(data, list) else []:
                s = it.get("symbol") or it.get("contractCode") or it.get("symbolName")
                if s == symbol:
                    pp = int(it.get("pricePrecision", it.get("pricePrecisionNum", 4)))
                    qp = int(it.get("quantityPrecision", it.get("volPrecision", 3)))
                    return max(pp, 0), max(qp, 0)
        except Exception as e:
            log(f"⚠️ get_symbol_filters fallback: {e}")
        return 4, 3

    # BingXClient 클래스 안에 추가
    def get_contract_spec(self, symbol: str) -> dict:
        """
        /openApi/swap/v2/quote/contracts에서 심볼 스펙을 읽어온다.
        반환:
        - contractSize: 1계약당 기초자산 수량
        - minQty: 최소 주문 수량
        - qtyStep: 수량 스텝(증감 단위)
        - pricePrecision, quantityPrecision
        """
        spec = {
            "contractSize": 1.0,
            "minQty": 0.0,
            "qtyStep": 1.0,
            "pricePrecision": 4,
            "quantityPrecision": 0,
        }
        try:
            url = f"{BASE}/openApi/swap/v2/quote/contracts"
            j = _req_get(url)
            data = j.get("data", [])
            if isinstance(data, list):
                for it in data:
                    s = it.get("symbol") or it.get("contractCode") or it.get("symbolName")
                    if s == symbol:
                        pp = int(it.get("pricePrecision") or it.get("pricePrecisionNum") or 4)
                        qp = int(it.get("quantityPrecision") or it.get("volPrecision") or 0)
                        contract_size = float(it.get("contractSize") or it.get("multiplier") or 1.0)
                        min_qty = float(it.get("minQty") or it.get("minVol") or it.get("minTradeNum") or 0.0)
                        qty_step = it.get("volumeStep") or it.get("stepSize")
                        if qty_step is None:
                            qty_step = 1.0 if qp == 0 else 10 ** (-qp)
                        spec.update({
                            "contractSize": contract_size,
                            "minQty": float(qty_step) if min_qty == 0 else float(min_qty),
                            "qtyStep": float(qty_step),
                            "pricePrecision": pp,
                            "quantityPrecision": qp,
                        })
                        return spec
        except Exception as e:
            log(f"⚠️ get_contract_spec fallback: {e}")
        return spec

    # ----- Account / Balance -----
    def get_available_usdt(self) -> float:
        """가용 USDT 마진. /user/balance 서명 GET."""
        url = f"{BASE}/openApi/swap/v2/user/balance"
        j = _req_get(url, {"recvWindow": 60000, "timestamp": _ts()}, signed=True)
        data = j.get("data", {})
        bal = data.get("balance", data)

        def pick_available(d: dict) -> float:
            for k in ("availableMargin", "availableBalance", "available"):
                if k in d:
                    try:
                        return float(d[k])
                    except Exception:
                        pass
            for k in ("balance", "equity"):
                if k in d:
                    try:
                        return float(d[k])
                    except Exception:
                        pass
            return 0.0

        if isinstance(bal, dict):
            if bal.get("asset") in (None, "USDT"):
                return pick_available(bal)
        elif isinstance(bal, list):
            for b in bal:
                if b.get("asset") == "USDT":
                    return pick_available(b)

        for k in ("availableMargin", "availableBalance", "available"):
            if k in data:
                return float(data[k])
        raise RuntimeError(f"unexpected /user/balance payload: {j}")

    def set_margin_mode(self, symbol: str, mode: str):
        """마진 모드 설정 - UI: CROSS / ISOLATED - API: CROSSED / ISOLATED"""
        m_primary = "CROSSED" if mode.upper().startswith("CROSS") else "ISOLATED"
        url = f"{BASE}/openApi/swap/v2/trade/marginType"
        body1 = {"symbol": symbol, "marginMode": m_primary, "recvWindow": 60000, "timestamp": _ts()}
        try:
            _req_post(url, body1, signed=True)
            return
        except Exception as e1:
            log(f"⚠️ set_margin_mode(primary) failed: {e1}")
        body2 = {"symbol": symbol, "marginType": m_primary, "recvWindow": 60000, "timestamp": _ts()}
        try:
            _req_post(url, body2, signed=True)
        except Exception as e2:
            log(f"⚠️ set_margin_mode(alt) failed: {e2}")
        if SKIP_SETUP:
            log("ℹ️ SKIP_SETUP=TRUE → set_margin_mode 생략")
            return
        # 반복 시도 (환경에 따라 첫 호출이 먹히지 않는 케이스 대비)
        try:
            _req_post(url, body1, signed=True)
        except Exception as e1:
            log(f"⚠️ set_margin_mode(primary,retry) failed: {e1}")
        try:
            _req_post(url, body2, signed=True)
        except Exception as e2:
            log(f"⚠️ set_margin_mode(alt,retry) failed: {e2}")

    def set_leverage(self, symbol: str, leverage: int):
        """레버리지 설정 - HEDGE 모드: LONG/SHORT 각각 호출 - ONEWAY 모드: LONG만 시도"""
        url = f"{BASE}/openApi/swap/v2/trade/leverage"
        sides = ["LONG", "SHORT"] if POSITION_MODE == "HEDGE" else ["LONG"]
        for s in sides:
            body1 = {
                "symbol": symbol,
                "side": s,
                "leverage": int(leverage),
                "recvWindow": 60000,
                "timestamp": _ts(),
            }
            try:
                _req_post(url, body1, signed=True)
                continue
            except Exception as e1:
                log(f"⚠️ set_leverage(primary,{s}) failed: {e1}")
            body2 = {
                "symbol": symbol,
                "positionSide": s,
                "leverage": int(leverage),
                "recvWindow": 60000,
                "timestamp": _ts(),
            }
            try:
                _req_post(url, body2, signed=True)
            except Exception as e2:
                log(f"⚠️ set_leverage(alt,{s}) failed: {e2}")
            if SKIP_SETUP:
                log("ℹ️ SKIP_SETUP=TRUE → set_leverage 생략")
                return
            # 반복 시도 (환경 차이 대응)
            try:
                _req_post(url, body1, signed=True)
            except Exception as e1:
                log(f"⚠️ set_leverage(primary,retry,{s}) failed: {e1}")
            try:
                _req_post(url, body2, signed=True)
            except Exception as e2:
                log(f"⚠️ set_leverage(alt,retry,{s}) failed: {e2}")

    # ----- Orders / Positions -----
    def place_market(self, symbol: str, side: str, qty: float,
                     reduce_only: bool = False, position_side: str | None = None,
                     close_position: bool = False) -> str:
        import math
        url = f"{BASE}/openApi/swap/v2/trade/order"

        # === 정밀도/최소수량/스텝 보정 ===
        pp, qp = self.get_symbol_filters(symbol)
        step = 1.0 if qp == 0 else 10 ** (-qp)
        try:
            spec = self.get_contract_spec(symbol)
            min_qty = float(spec.get("tradeMinQuantity") or spec.get("minQty") or spec.get("minVol") or spec.get("minTradeNum") or 0.0)
            step    = float(spec.get("qtyStep") or spec.get("volumeStep") or spec.get("stepSize") or step)
        except Exception:
            min_qty = 0.0

        qty = max(qty, 0.0)
        qty = math.floor(qty / step) * step
        qty = float(f"{qty:.{max(qp,0)}f}")
        if qty < (min_qty or step):
            qty = (min_qty or step)
        if qty <= 0:
            raise RuntimeError(f"quantity <= 0 after adjust (qp={qp}, min={min_qty}, step={step})")

        base = {
            "symbol": symbol,
            "type": "MARKET",
            "side": side.upper(),
            "quantity": qty,
            "recvWindow": 60000,
            "timestamp": _ts(),
        }

        if POSITION_MODE == "HEDGE":
            # ✅ 항상 넣기 (기본값 LONG/SHORT)
            ps = (position_side or ("LONG" if side.upper()=="BUY" else "SHORT")).upper()
            base["positionSide"] = ps
            # HEDGE에서는 reduceOnly는 넣지 않음 (109400 방지)
        else:
            if reduce_only:
                base["reduceOnly"] = True

        # close 포지션용 변형(필요시)
        variants = []
        if close_position:
            v = dict(base)
            v["closePosition"] = "true"   # ← 문자열
            variants.append(v)
        variants.append(base)

        return self._try_order(url, variants)

    def place_limit(self, symbol: str, side: str, qty: float, price: float,
                    reduce_only: bool = False, position_side: str | None = None,
                    tif: str = "GTC", close_position: bool = False) -> str:
        import math
        url = f"{BASE}/openApi/swap/v2/trade/order"

        # === 정밀도/최소수량/스텝 보정 ===
        pp, qp = self.get_symbol_filters(symbol)
        step = 1.0 if qp == 0 else 10 ** (-qp)
        try:
            spec = self.get_contract_spec(symbol)
            min_qty = float(spec.get("tradeMinQuantity") or spec.get("minQty") or spec.get("minVol") or spec.get("minTradeNum") or 0.0)
            step    = float(spec.get("qtyStep") or spec.get("volumeStep") or spec.get("stepSize") or step)
        except Exception:
            min_qty = 0.0

        qty = max(qty, 0.0)
        qty = math.floor(qty / step) * step
        qty = float(f"{qty:.{max(qp,0)}f}")
        if qty < (min_qty or step):
            qty = (min_qty or step)
        if qty <= 0:
            raise RuntimeError(f"quantity <= 0 (limit) after adjust (qp={qp}, min={min_qty}, step={step})")

        price = float(f"{float(price):.{max(pp,0)}f}")
        if price <= 0:
            raise RuntimeError("price <= 0 (limit)")

        base = {
            "symbol": symbol,
            "type": "LIMIT",
            "side": side.upper(),
            "quantity": qty,
            "price": price,
            "timeInForce": tif,
            "recvWindow": 60000,
            "timestamp": _ts(),
        }

        if POSITION_MODE == "HEDGE":
            ps = (position_side or ("LONG" if side.upper()=="BUY" else "SHORT")).upper()
            base["positionSide"] = ps
            # HEDGE에서는 reduceOnly는 넣지 않음
        else:
            if reduce_only:
                base["reduceOnly"] = True

        variants = []
        if close_position:
            v = dict(base)
            v["closePosition"] = "true"   # ← 문자열
            variants.append(v)
        variants.append(base)

        return self._try_order(url, variants)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        url = f"{BASE}/openApi/swap/v2/trade/cancel"  # ← 보정
        try:
            _req_delete(url, {"symbol": symbol, "orderId": int(order_id),
                              "recvWindow": 60000, "timestamp": _ts()}, signed=True)
            return True
        except Exception as e:
            log(f"⚠️ cancel_order: {e}")
            return False

    def open_orders(self, symbol: str) -> list[dict]:
        """열려있는 주문 목록"""
        url = f"{BASE}/openApi/swap/v2/trade/openOrders"
        try:
            j = _req_get(url, {"symbol": symbol, "recvWindow": 60000, "timestamp": _ts()}, signed=True)
            data = j.get("data", [])
            return data if isinstance(data, list) else (data.get("orders", []) if isinstance(data, dict) else [])
        except Exception as e:
            log(f"⚠️ open_orders: {e}")
            return []

    def position_info(self, symbol: str, side: str) -> tuple[float, float]:
        """현재 포지션 (평단가, 수량). 없으면 (0.0, 0.0)"""
        url = f"{BASE}/openApi/swap/v2/user/positions"
        j = _req_get(url, {"symbol": symbol, "recvWindow": 60000, "timestamp": _ts()}, signed=True)
        arr = j.get("data", [])
        want = "LONG" if str(side).upper() == "BUY" else "SHORT"

        pos = None
        for p in arr if isinstance(arr, list) else []:
            if POSITION_MODE == "HEDGE":
                ps = (p.get("positionSide") or p.get("posSide") or p.get("side") or "").upper()
                if ps == want:
                    pos = p
                    break
            else:
                pos = p
                break

        if not pos:
            return 0.0, 0.0

        entry_keys = ["entryPrice", "avgPrice", "avgEntryPrice", "openPrice", "positionOpenPrice"]
        entry = 0.0
        for k in entry_keys:
            v = pos.get(k)
            if v not in (None, ""):
                try:
                    entry = float(v)
                    if entry > 0:
                        break
                except Exception:
                    pass

        qty_keys = ["positionAmt", "positionAmount", "quantity", "positionQty", "positionSize", "amount", "qty"]
        qty = 0.0
        for k in qty_keys:
            v = pos.get(k)
            if v not in (None, ""):
                try:
                    qty = abs(float(v))
                    if qty > 0:
                        break
                except Exception:
                    pass

        return entry, qty
