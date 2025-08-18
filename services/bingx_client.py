# services/bingx_client.py
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from utils.logging import log
from urllib3.util.retry import Retry


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


# ---- 타임아웃/재시도 설정(환경변수) ----
CONN_TIMEOUT = float(os.getenv("HTTP_CONN_TIMEOUT", "5"))    # 연결 타임아웃
READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "60"))   # 응답(읽기) 타임아웃
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))   # 자동 재시도 횟수
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "0.5"))       # 재시도 지수 백오프 계수

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=HTTP_MAX_RETRIES,
        read=HTTP_MAX_RETRIES,
        connect=HTTP_MAX_RETRIES,
        status=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "DELETE"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = _build_session()


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

# --- normalize payload for signed requests (bool -> "true"/"false") ---
def _coerce_params(d: dict | None) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        elif v is None:
            # 서명 파라미터에 None은 넣지 않음
            continue
        else:
            out[k] = v
    return out

def _do_request(method: str, url: str, *, params=None, data=None, headers=None, signed=False, form=False):
    timeout = (CONN_TIMEOUT, READ_TIMEOUT)
    try:
        if method == "GET":
            return SESSION.get(url, params=params, headers=headers, timeout=timeout)
        elif method == "POST":
            if form:
                return SESSION.post(url, data=data, headers=headers, timeout=timeout)
            else:
                return SESSION.post(url, json=data, headers=headers, timeout=timeout)
        elif method == "DELETE":
            return SESSION.delete(url, params=params, headers=headers, timeout=timeout)
        else:
            raise RuntimeError(f"unsupported method: {method}")
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError(f"ReadTimeout @{method} {url}: {e}")
    except requests.exceptions.ConnectTimeout as e:
        raise RuntimeError(f"ConnectTimeout @{method} {url}: {e}")
    except requests.exceptions.ConnectionError as e:
        # 원격 종료/일시 네트워크도 여기로 들어옴
        raise RuntimeError(f"ConnectionError @{method} {url}: {e}")



def _req_get(url: str, params: dict | None = None, signed: bool = False) -> dict:
    params = params or {}
    if signed:
        qs = _sign(params)
        r = _do_request("GET", url + "?" + qs, headers=_headers(), signed=True)
    else:
        r = _do_request("GET", url, params=params, headers=_headers())
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


def _req_post(url: str, body: dict | None = None, signed: bool = False) -> dict:
    body = body or {}
    if signed:
        payload = _sign(body)  # qs + signature
        r = _do_request("POST", url, data=payload, headers=_headers(form=True), signed=True, form=True)
    else:
        r = _do_request("POST", url, data=body, headers=_headers(form=False), form=False)
    j = r.json()
    code = str(j.get("code", "0"))
    if code != "0":
        raise RuntimeError(f"BingX error @POST {url}: {code} {j.get('msg')}")
    return j

def _req_delete(url: str, params: dict | None = None, signed: bool = False) -> dict:
    params = params or {}
    if signed:
        qs = _sign(params)
        r = _do_request("DELETE", url + "?" + qs, headers=_headers(form=True), signed=True)
    else:
        r = _do_request("DELETE", url, params=params, headers=_headers(form=False))
    j = r.json()
    code = str(j.get("code", "0"))
    if code != "0":
        raise RuntimeError(f"BingX error @DELETE {url}: {code} {j.get('msg')}")
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
        """
        BingX 주문 응답에서 orderId를 최대한 유연하게 추출한다.
        지원 케이스:
        - resp["orderId"] / resp["id"]
        - resp["data"]["orderId"] / resp["data"]["id"]
        - resp["data"]["order"]["orderId"] / ["orderID"] / ["id"]
        """
        if not isinstance(resp, dict):
            return ""

        # 최상위
        for k in ("orderId", "orderID", "id"):
            v = resp.get(k)
            if v:
                return str(v)

        d = resp.get("data")
        if isinstance(d, dict):
            # data 바로 아래
            for k in ("orderId", "orderID", "id"):
                v = d.get(k)
                if v:
                    return str(v)
            # data.order 아래
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
        last_err = None
        for body in variants:
            try:
                j = _req_post(url, body, signed=True)
                oid = self._extract_order_id(j)   # ✅ 항상 폭넓게 추출
                if oid:
                    return str(oid)
                last_err = RuntimeError(f"missing orderId in response: {j}")
            except Exception as e:
                last_err = e
                log(f"⚠️ order variant failed: {e}")
                continue
        raise last_err or RuntimeError("all order variants failed")
    
    def cancel_open_orders(self, symbol: str, side: str|None=None,
                        keep_close_position: bool=True) -> int:
        """
        열려있는 주문을 취소.
        - side가 주어지면 그 side(BUY/SELL)만
        - keep_close_position=True면 closePosition=true 인 주문(TP)은 건드리지 않음
        반환: 취소 시도한 개수
        """
        oo = self.open_orders(symbol)
        count = 0
        for o in oo:
            try:
                if side:
                    s = (o.get("side") or "").upper()
                    if s != side.upper():
                        continue
                if keep_close_position:
                    cp = str(o.get("closePosition") or "").lower()
                    if cp == "true" or cp is True:
                        continue
                oid = str(o.get("orderId") or o.get("orderID") or o.get("id") or "")
                if oid:
                    self.cancel_order(symbol, oid)
                    count += 1
            except Exception:
                # 개별 실패는 무시하고 계속
                pass
        return count


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
    def place_market(self, symbol: str, side: str, qty: float,
                    reduce_only: bool=False, position_side: str|None=None,
                    close_position: bool=False) -> str:
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
            "side": side.upper(),             # BUY/SELL
            "quantity": qty,
            "recvWindow": 60000,
            "timestamp": _ts(),
        }

        if POSITION_MODE == "HEDGE":
            # ✅ 항상 넣기 (기본값 LONG/SHORT)
            ps = (position_side or ("LONG" if side.upper()=="BUY" else "SHORT")).upper()
            base["positionSide"] = ps
            # HEDGE에서는 reduceOnly는 넣지 않는 편이 안전
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
                    reduce_only: bool=False, position_side: str|None=None,
                    tif: str="GTC", close_position: bool=False) -> str:
        import math
        url = f"{BASE}/openApi/swap/v2/trade/order"

        # === 정밀도/최소수량/스텝 보정 ===
        pp, qp = self.get_symbol_filters(symbol)
        min_qty = 0.0
        step = 1.0 if qp == 0 else 10 ** (-qp)
        try:
            spec = self.get_contract_spec(symbol)
            min_qty = float(spec.get("tradeMinQuantity") or spec.get("minQty") or spec.get("minVol") or spec.get("minTradeNum") or 0.0)
            step = float(spec.get("qtyStep") or spec.get("volumeStep") or spec.get("stepSize") or step)
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

        # HEDGE: 반드시 positionSide, reduceOnly 금지
        if POSITION_MODE == "HEDGE":
            base["positionSide"] = position_side or ("LONG" if side.upper()=="BUY" else "SHORT")
        else:
            if reduce_only:
                base["reduceOnly"] = True

        # ✅ 변형 시도: (1) closePosition="true"(문자열) → (2) 일반 LIMIT
        variants = []
        if close_position:
            v = dict(base)
            v["closePosition"] = "true"   # ✅ 문자열 "true" 필수
            variants.append(v)
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
