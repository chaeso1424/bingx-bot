from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_UP

def calc_tp_price(entry_price, side, tp_percent, leverage=None):
    """
    Calculate the target take-profit price.

    :param entry_price: Entry (average) price of the position.
    :param side: "BUY" for long positions or "SELL" for short positions.
    :param tp_percent: Target percentage return on investment.
    :param leverage: Ignored for backwards compatibility.
    :return: Target price at which to place a take-profit order.
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
    Compute a new weighted average price when adding a fill to an existing position.

    :param prev_avg: Previous average price.
    :param prev_qty: Previous quantity.
    :param fill_price: Execution price of the fill.
    :param fill_qty: Quantity of the fill.
    :return: Tuple of (new_avg, new_qty) rounded to a reasonable number of decimals.
    """
    if fill_qty <= 0:
        return prev_avg, prev_qty
    new_qty = prev_qty + fill_qty
    if new_qty <= 0:
        return 0.0, 0.0
    new_avg = (prev_avg * prev_qty + fill_price * fill_qty) / new_qty
    return round(new_avg, 6), round(new_qty, 6)


def floor_to_step(value: float, step: float) -> float:
    """
    Round ``value`` down to the nearest multiple of ``step``.

    If ``step`` is zero or negative, the original value is returned.
    """
    if step <= 0:
        return value
    import math
    return math.floor(value / step) * step


def _safe_close_qty(qty_now: float, step: float, min_allowed: float) -> float:
    from decimal import Decimal, getcontext, ROUND_FLOOR
    getcontext().prec = 28

    if step <= 0:
        return 0.0

    qd  = Decimal(str(qty_now))   # float 오차 방지
    sd  = Decimal(str(step))
    mind = Decimal(str(min_allowed))

    # 최소단위 미만이면 스킵
    if qd < mind:
        return 0.0

    # step 기준 스냅 (반올림 보정)
    remainder = qd % sd
    if remainder <= (sd / 2):
        qd = qd - remainder
    else:
        qd = qd - remainder + sd

    # 혹시라도 위쪽으로 스냅했는데 포지션 초과 방지 → 다시 floor
    n = (qd / sd).to_integral_value(rounding=ROUND_FLOOR)
    q = n * sd

    if q < mind:
        return 0.0

    return float(q)


def _round_to_tick(value: float, pp: int, mode: str = "DOWN") -> float:
    """
    Round ``value`` to a price tick defined by ``pp`` (price precision).

    :param value: Price value to round.
    :param pp: Price precision (number of decimal places allowed).
    :param mode: "DOWN", "UP", or "NEAREST".
    :return: The rounded price.
    """
    tick = Decimal('1').scaleb(-int(pp))  # tick = 10^-pp
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
    Calculate a take-profit price based on desired ROI and leverage.

    :param entry_price: The entry (avg) price.
    :param side: "BUY" for long positions or "SELL" for short positions.
    :param roi_percent: Desired return on investment percentage.
    :param leverage: Leverage factor (ignored if less than 1.0).
    :param pp: Price precision (number of decimal places).
    :return: The take-profit price rounded appropriately so as not to exceed the desired ROI.
    """
    side_u = str(side).upper()
    lev = max(float(leverage), 1.0)
    base = float(entry_price)
    roi = float(roi_percent) / 100.0
    if base <= 0 or roi <= 0:
        return 0.0
    # Price change percentage = ROI / leverage
    price_pct = roi / lev
    ideal = base * (1.0 + price_pct) if side_u == "BUY" else base * (1.0 - price_pct)
    # LONG rounds down (avoid overshooting ROI); SHORT rounds up
    mode = "DOWN" if side_u == "BUY" else "UP"
    return _round_to_tick(ideal, int(pp), mode)
