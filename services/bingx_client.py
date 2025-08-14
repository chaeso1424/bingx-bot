# services/bingx_client.py
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from utils.logging import log

import os
# --- force IPv4 only for urllib3/requests ---
import socket
try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

SKIP_SETUP = os.getenv("SKIP_SETUP", "false").lower() == "true"

API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
BASE = os.getenv("BINGX_BASE", "https://open-api.bingx.com")
POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "HEDGE").upper()  # HEDGE or ONEWAY




# ---------- low-level utils ----------
def _ts():
    return int(time.time() * 1000)


def _headers(form: bool = False):
    if not API_KEY:
        raise RuntimeError("BINGX_API_KEY is empty. Check your .env and app load order.")
    h = {"X-BX-APIKEY": API_KEY}
    h["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
    return h

def _sign(params: dict) -> str:
    """
    Return "query_string&signature=..." for signed requests.
    """
    qs = urlencode(params)
    sig = hmac.new(API_SECRET.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig


def _req_get(url: str, params: dict | None = None, signed: bool = False) -> dict:
    params = params or {}
    if signed:
        qs = _sign(params)
        r = requests.get(url + "?" + qs, headers=_headers(), timeout=(3, 30))
    else:
        r = requests.get(url, params=params, headers=_headers(), timeout=(3, 30))
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
        qs = _sign(params)
        r = requests.delete(url + "?" + qs, headers=_headers(form=True), timeout=(3, 30))
    else:
        r = requests.delete(url, params=params, headers=_headers(form=False), timeout=(3, 30))
    j = r.json()
    code = str(j.get("code", "0"))
    if code != "0":
        raise RuntimeError(f"BingX error @DELETE {url}: {code} {j.get('msg')}")
    return j

def _req_post(url: str, body: dict | None = None, signed: bool = False) -> dict:
    body = body or {}
    if signed:
        payload = _sign(body)  # querystring + signature
        r = requests.post(url, data=payload, headers=_headers(form=True), timeout=(3, 30))
    else:
        r = requests.post(url, json=body, headers=_headers(form=False), timeout=(3, 30))
    j = r.json()
    code = str(j.get("code", "0"))
    if code != "0":
        raise RuntimeError(f"BingX error @POST {url}: {code} {j.get('msg')}")
    return j



# ---------- high-level client ----------
class BingXClient:

    def get_current_leverage(self, symbol: str, side: str) -> float | None:
        """
        현재 심볼/사이드의 적용 레버리지(거래소 측 상태)를 조회.
        HEDGE 모드면 positionSide별로 조회.
        """
        url = f"{BASE}/openApi/swap/v2/user/positions"
        j = _req_get(url, {"symbol": symbol, "recvWindow": 60000, "timestamp": _ts()}, signed=True)
        arr = j.get("data", [])
        want = "LONG" if str(side).upper() == "BUY" else "SHORT"
        for p in arr if isinstance(arr, list) else []:
            if POSITION_MODE == "HEDGE":
                ps = (p.get("positionSide") or p.get("posSide") or p.get("side") or "").upper()
                if ps != want:
                    continue
            # 키 이름 다양성 대비
            lev = p.get("leverage") or p.get("leverageLevel") or p.get("positionLeverage")
            if lev is not None:
                try:
                    return float(lev)
                except:
                    pass
        return None

    def __init__(self):
        self._spec_cache = {}

    # (없으면 추가) 주문 응답에서 orderId 추출
    def _extract_order_id(self, resp: dict) -> str:
        for k in ("orderId", "orderID", "id"):
            if k in resp and resp[k]:
                return str(resp[k])
        d = resp.get("data", {})
        if isinstance(d, dict):
            for k in ("orderId", "orderID", "id"):
                if k in d and d[k]:
                    return str(d[k])
            o = d.get("order")
            if isinstance(o, dict):
                for k in ("orderId", "orderID", "id", "order_id"):
                    if k in o and o[k]:
                        return str(o[k])
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
                    return max(pp,0), max(qp,0)
        except Exception as e:
            log(f"⚠️ get_symbol_filters fallback: {e}")
        return 4, 3

    def _round_to_precision(self, value: float, digits: int) -> float:
        if digits < 0: digits = 0
        fmt = f"{{:.{digits}f}}"
        return float(fmt.format(value))

    
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
            j = _req_get(url)  # unsigned
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



    def _try_order(self, url: str, variants: list[dict]) -> str:
        """
        variants를 순차 시도. 성공하면 orderId 반환.
        """
        last_err = None
        for body in variants:
            try:
                j = _req_post(url, body, signed=True)
                data = j.get("data", j)
                oid = (isinstance(data, dict) and (data.get("orderId") or data.get("id"))) or j.get("orderId")
                if oid:
                    return str(oid)
                last_err = RuntimeError(f"missing orderId in response: {j}")
            except Exception as e:
                last_err = e
                log(f"⚠️ order variant failed: {e}")
                continue
        raise last_err or RuntimeError("all order variants failed")

    # ----- Market / Quote -----
    def list_symbols(self) -> list[str]:
        """
        거래가능 선물 심볼 목록. 실패시 안전 기본값 반환.
        """
        try:
            url = f"{BASE}/openApi/swap/v2/quote/contracts"
            j = _req_get(url)  # unsigned
            data = j.get("data", [])
            out = []
            for it in data if isinstance(data, list) else []:
                s = it.get("symbol") or it.get("contractCode") or it.get("symbolName")
                if s and s.endswith("USDT"):
                    out.append(s)
            if out:
                return sorted(set(out))
        except Exception as e:
            log(f"⚠️ list_symbols fallback: {e}")
        return ["BTC-USDT", "ETH-USDT", "DOGE-USDT"]

    def get_last_price(self, symbol: str) -> float:
        """
        최신 체결가
        """
        url = f"{BASE}/openApi/swap/v2/quote/price"
        j = _req_get(url, {"symbol": symbol})
        d = j.get("data", {})
        return float(d.get("price"))

    def get_mark_price(self, symbol: str) -> float:
        """
        마크프라이스 (없을 경우 최신가로 폴백)
        """
        try:
            url = f"{BASE}/openApi/swap/v2/quote/premiumIndex"
            j = _req_get(url, {"symbol": symbol})
            d = j.get("data", {})
            for k in ("markPrice", "indexPrice", "price"):
                if k in d:
                    return float(d[k])
        except Exception as e:
            log(f"⚠️ get_mark_price fallback to last price: {e}")
        return self.get_last_price(symbol)

    # ----- Account / Balance -----
    def get_available_usdt(self) -> float:
        """
        가용 USDT 마진. /user/balance 서명 GET.
        """
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
            # 단일 자산 객체
            if bal.get("asset") in (None, "USDT"):
                return pick_available(bal)
        elif isinstance(bal, list):
            # 여러 자산 배열
            for b in bal:
                if b.get("asset") == "USDT":
                    return pick_available(b)

        # 최후 방어
        for k in ("availableMargin", "availableBalance", "available"):
            if k in data:
                return float(data[k])
        raise RuntimeError(f"unexpected /user/balance payload: {j}")
        
    

    # ----- Settings (Margin mode / Leverage) -----
    def set_margin_mode(self, symbol: str, mode: str):
        """
        마진 모드 설정
        - UI: CROSS / ISOLATED
        - API: CROSSED / ISOLATED
        일부 환경에서 marginMode 대신 marginType 키를 요구하기도 하므로 이중 시도.
        """
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
        m_primary = "CROSSED" if mode.upper().startswith("CROSS") else "ISOLATED"
        url = f"{BASE}/openApi/swap/v2/trade/marginType"
        body1 = {"symbol": symbol, "marginMode": m_primary, "recvWindow": 60000, "timestamp": _ts()}
        try:
            _req_post(url, body1, signed=True)
            return
        except Exception as e1:
            log(f"⚠️ set_margin_mode(primary,{url}) failed: {e1}")
        body2 = {"symbol": symbol, "marginType": m_primary, "recvWindow": 60000, "timestamp": _ts()}
        try:
            _req_post(url, body2, signed=True)
        except Exception as e2:
            # 여기서도 실패해도 '진행'하도록 바꿈 (강제 종료 X)
            log(f"⚠️ set_margin_mode(alt,{url}) failed: {e2}")

    def set_leverage(self, symbol: str, leverage: int):
        """
        레버리지 설정
        - HEDGE 모드: LONG/SHORT 각각 호출
        - ONEWAY 모드: LONG만 시도(백엔드에서 BOTH가 필요 없는 케이스가 있음)
        일부 환경에서 side 대신 positionSide 키로 받아들이기도 하므로 이중 시도.
        """
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
            url = f"{BASE}/openApi/swap/v2/trade/leverage"
            sides = ["LONG", "SHORT"] if POSITION_MODE == "HEDGE" else ["LONG"]
            for s in sides:
                body1 = {"symbol": symbol, "side": s, "leverage": int(leverage), "recvWindow": 60000, "timestamp": _ts()}
                try:
                    _req_post(url, body1, signed=True)
                    continue
                except Exception as e1:
                    log(f"⚠️ set_leverage(primary,{s},{url}) failed: {e1}")
                body2 = {"symbol": symbol, "positionSide": s, "leverage": int(leverage), "recvWindow": 60000, "timestamp": _ts()}
                try:
                    _req_post(url, body2, signed=True)
                except Exception as e2:
                    # 실패해도 '진행'하도록 바꿈
                    log(f"⚠️ set_leverage(alt,{s},{url}) failed: {e2}")

    # ----- Orders / Positions -----
    def place_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        position_side: str | None = None,
        close_position: bool = False,    # ✅ 추가
    ) -> str:
        import math
        url = f"{BASE}/openApi/swap/v2/trade/order"

        pp, qp = self.get_symbol_filters(symbol)
        min_qty = 0.0
        step = 1.0 if qp == 0 else 10 ** (-qp)
        try:
            spec = self.get_contract_spec(symbol)
            min_qty = float(spec.get("tradeMinQuantity") or spec.get("minQty") or spec.get("minVol") or spec.get("minTradeNum") or 0.0)
            step    = float(spec.get("qtyStep") or spec.get("volumeStep") or spec.get("stepSize") or step)
        except Exception:
            pass

        qty = max(qty, 0.0)
        qty = math.floor(qty / step) * step
        qty = float(f"{qty:.{max(qp,0)}f}")
        if qty < (min_qty or step):
            qty = (min_qty or step)
        if qty <= 0:
            raise RuntimeError(f"quantity <= 0 after precision adjust (qp={qp}, min={min_qty}, step={step})")

        base = {
            "symbol": symbol,
            "type": "MARKET",
            "side": side.upper(),
            "quantity": qty,
            "recvWindow": 60000,
            "timestamp": _ts(),
        }

        variants = []
        if POSITION_MODE == "HEDGE":
            ps = position_side or ("LONG" if side.upper() == "BUY" else "SHORT")
            base["positionSide"] = ps
            if close_position:
                # market closePosition는 보통 quantity 없이도 허용
                v_noqty = {k: v for k, v in base.items() if k != "quantity"}
                v_noqty["closePosition"] = True
                variants.append(v_noqty)
                variants.append({**base, "closePosition": True})
            variants.append(base)
        else:
            if close_position or reduce_only:
                variants.append({**base, "reduceOnly": True})
            variants.append(base)

        return self._try_order(url, variants)





    def place_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool = False,
        position_side: str | None = None,
        close_position: bool = False,    # ✅ 추가
        tif: str = "GTC",
    ) -> str:
        import math
        url = f"{BASE}/openApi/swap/v2/trade/order"

        # === 정밀도/최소수량 보정 ===
        pp, qp = self.get_symbol_filters(symbol)
        min_qty = 0.0
        step = 1.0 if qp == 0 else 10 ** (-qp)
        try:
            spec = self.get_contract_spec(symbol)
            min_qty = float(spec.get("tradeMinQuantity") or spec.get("minQty") or spec.get("minVol") or spec.get("minTradeNum") or 0.0)
            step    = float(spec.get("qtyStep") or spec.get("volumeStep") or spec.get("stepSize") or step)
        except Exception:
            pass

        qty = max(qty, 0.0)
        qty = math.floor(qty / step) * step
        qty = float(f"{qty:.{max(qp,0)}f}")
        if qty < (min_qty or step):
            qty = (min_qty or step)
        if qty <= 0:
            raise RuntimeError(f"quantity <= 0 (tp/limit) after adjust (qp={qp}, min={min_qty}, step={step})")

        price = float(f"{float(price):.{max(pp,0)}f}")
        if price <= 0:
            raise RuntimeError("price <= 0 (tp/limit)")

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

        variants = []

        if POSITION_MODE == "HEDGE":
            ps = position_side or ("LONG" if side.upper() == "BUY" else "SHORT")
            base["positionSide"] = ps

            if close_position:
                # ① 닫기전용 + 수량 제거 (일부 환경은 closePosition일 때 quantity 금지)
                v_noqty = {k: v for k, v in base.items() if k != "quantity"}
                v_noqty["closePosition"] = True
                variants.append(v_noqty)

                # ② 닫기전용 + 수량 포함 (반대로 수량을 요구하는 환경)
                variants.append({**base, "closePosition": True})

            # ③ 일반 LIMIT
            variants.append(base)

            # ④ 호환형 키 (orderType/qty)
            variants.append({**base, "orderType": "LIMIT", "quantity": None, "qty": qty})

        else:
            # ONEWAY: 닫기전용 효과는 reduceOnly
            if close_position or reduce_only:
                variants.append({**base, "reduceOnly": True})
            variants.append(base)

        return self._try_order(url, variants)





    def cancel_order(self, symbol: str, order_id: str) -> bool:
        url = f"{BASE}/openApi/swap/v2/trade/order"
        try:
            _req_delete(url, {"symbol": symbol, "orderId": int(order_id),
                            "recvWindow": 60000, "timestamp": _ts()}, signed=True)
            return True
        except Exception as e:
            log(f"⚠️ cancel_order: {e}")
            return False


    def open_orders(self, symbol: str) -> list[dict]:
        """
        열려있는 주문 목록
        """
        url = f"{BASE}/openApi/swap/v2/trade/openOrders"
        try:
            j = _req_get(url, {"symbol": symbol, "recvWindow": 60000, "timestamp": _ts()}, signed=True)
            data = j.get("data", [])
            return data if isinstance(data, list) else data.get("orders", [])
        except Exception as e:
            log(f"⚠️ open_orders: {e}")
            return []

    def position_info(self, symbol: str, side: str) -> tuple[float, float]:
        """
        현재 포지션 (평단가, 수량). 없으면 (0.0, 0.0)
        - 다양한 키 이름을 폭넓게 지원해서 avg가 0으로 들어오는 걸 최소화
        """
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

        # 평균가 후보
        entry_keys = ["entryPrice", "avgPrice", "avgEntryPrice", "openPrice", "positionOpenPrice"]
        entry = 0.0
        for k in entry_keys:
            v = pos.get(k)
            if v not in (None, ""):
                try:
                    entry = float(v)
                    if entry > 0:
                        break
                except:
                    pass

        # 수량 후보
        qty_keys = ["positionAmt", "positionAmount", "quantity", "positionQty", "positionSize", "amount", "qty"]
        qty = 0.0
        for k in qty_keys:
            v = pos.get(k)
            if v not in (None, ""):
                try:
                    qty = abs(float(v))
                    if qty > 0:
                        break
                except:
                    pass

        return entry, qty
