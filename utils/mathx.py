def calc_tp_price(entry_price, side, tp_percent, leverage=None):
    """
    TP 목표가 계산.
    - entry_price: float (평단 또는 마크가)
    - side: "BUY" or "SELL"
    - tp_percent: 퍼센트 (예: 4.5)
    - leverage: (하위호환용) 무시됨
    """
    try:
        base = float(entry_price)
        pct = float(tp_percent) / 100.0
    except Exception:
        return 0.0
    if base <= 0 or pct <= 0:
        return 0.0
    return base * (1.0 + pct) if str(side).upper() == "BUY" else base * (1.0 - pct)


def weighted_avg_price(prev_avg: float, prev_qty: float, fill_price: float, fill_qty: float):
    """
    새 체결이 합산될 때 평단 갱신
    """
    if fill_qty <= 0:
        return prev_avg, prev_qty
    new_qty = prev_qty + fill_qty
    if new_qty <= 0:
        return 0.0, 0.0
    new_avg = (prev_avg*prev_qty + fill_price*fill_qty) / new_qty
    return round(new_avg, 6), round(new_qty, 6)

def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    import math
    return math.floor(value / step) * step

from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_UP

def _round_to_tick(value: float, pp: int, mode: str = "DOWN") -> float:
    """
    value를 가격 정밀도(pp)의 틱(=10^-pp)에 맞춰 반올림.
    mode: "DOWN" | "UP" | "NEAREST"
    """
    tick = Decimal('1').scaleb(-int(pp))   # 10^-pp
    v = Decimal(str(value)) / tick
    if mode == "DOWN":
        v = v.to_integral_value(rounding=ROUND_FLOOR)
    elif mode == "UP":
        v = v.to_integral_value(rounding=ROUND_CEILING)
    else:
        v = v.to_integral_value(rounding=ROUND_HALF_UP)
    return float(v * tick)

def tp_price_from_roi(entry_price: float, side: str, roi_percent: float, leverage: float, pp: int) -> float:
    """
    레버리지 ROI(%) 목표를 만족하는 TP 가격 계산.
    - LONG(=BUY 진입, SELL로 청산): 틱에 '내림' → 목표 ROI를 초과하지 않도록 보수적으로 설정
    - SHORT(=SELL 진입, BUY로 청산): 틱에 '올림' → 목표 ROI를 초과하지 않도록 보수적으로 설정
    """
    side_u = str(side).upper()
    lev = max(float(leverage), 1.0)
    base = float(entry_price)
    roi = float(roi_percent) / 100.0

    if base <= 0 or roi <= 0:
        return 0.0

    # 가격 변동률 = ROI / 레버리지
    price_pct = roi / lev
    ideal = base * (1.0 + price_pct) if side_u == "BUY" else base * (1.0 - price_pct)

    # ROI 과대 설정 방지 라운딩: LONG→DOWN, SHORT→UP
    mode = "DOWN" if side_u == "BUY" else "UP"
    return _round_to_tick(ideal, int(pp), mode)
