# runner.py
import threading, time
from utils.logging import log
from utils.mathx import tp_price_from_roi, floor_to_step
from models.config import BotConfig
from models.state import BotState
from services.bingx_client import BingXClient
import os

# ===== 운영 파라미터 =====
RESTART_DELAY_SEC   = int(os.getenv("RESTART_DELAY_SEC", "60"))   # TP 후 다음 사이클 대기
CLOSE_ZERO_STREAK   = int(os.getenv("CLOSE_ZERO_STREAK", "3"))    # 종료 판정에 필요한 연속 0회수
ZERO_EPS_FACTOR     = float(os.getenv("ZERO_EPS_FACTOR", "0.5"))  # 0 판정 여유(최소단위의 50%)
POLL_SEC            = 1.5

class BotRunner:
    def __init__(self, cfg: BotConfig, state: BotState, client: BingXClient):
        self.cfg = cfg
        self.state = state
        self.client = client
        self._thread = None
        self._stop = False

        # TP 기준값(모니터링 데드밴드용)
        self._last_tp_price = None
        self._last_tp_qty   = None

    # ---------- lifecycle ----------
    def start(self):
        if self.state.running:
            log("ℹ️ 이미 실행 중")
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.state.running = True
        self._thread.start()

    def stop(self):
        self._stop = True

    # ---------- helpers ----------
    def _now(self) -> float:
        return time.time()

    def _wait_cancel(self, order_id: str, timeout=3.0) -> bool:
        """취소 요청 후 openOrders에서 사라질 때까지 잠깐 대기."""
        t0 = time.time()
        want = str(order_id)
        while time.time() - t0 < timeout:
            try:
                oo = self.client.open_orders(self.cfg.symbol)
                alive = any(str(o.get("orderId") or o.get("orderID") or o.get("id") or "") == want for o in oo)
                if not alive:
                    return True
            except:
                pass
            time.sleep(0.2)
        return False

    def _estimate_required_margin(self, side: str, mark: float, spec: dict, pp: int, step: float) -> tuple[float, list]:
        """
        현재 설정(DCA, 레버리지)을 기준으로 '지금 가격'에서 모든 진입 주문(1차+리밋)에
        필요한 총 증거금(USDT)을 추정. (TP 제외)
        반환: (required_total_usdt, plan_list[{'type','price','qty','usdt'}...])
        """
        contract = float(spec.get("contractSize", 1.0)) or 1.0
        min_qty = float(spec.get("minQty", 0.0))
        lev = max(float(self.cfg.leverage), 1.0)

        def _plan_unit(price: float, usdt_amt: float) -> tuple[float, float, float]:
            target_notional = float(usdt_amt) * lev
            raw_qty = target_notional / max(price * contract, 1e-12)
            q = floor_to_step(raw_qty, step)
            # 거래소 최소 충족(축소가 아님, 필수 보정)
            if q < (min_qty or step):
                q = max(min_qty, step)
            need_margin = (q * price * contract) / lev
            return q, price, need_margin

        plan = []

        # 1차(시장가)
        first_usdt = float(self.cfg.dca_config[0][1])
        q1, p1, m1 = _plan_unit(mark, first_usdt)
        plan.append({"type": "MARKET", "price": p1, "qty": q1, "usdt": m1})

        # DCA 리밋들
        cum = 0.0
        for gap, usdt_amt in self.cfg.dca_config[1:]:
            cum += float(gap)
            price = mark * (1 - cum/100.0) if side == "BUY" else mark * (1 + cum/100.0)
            price = float(f"{price:.{pp}f}")
            q, p, m = _plan_unit(price, float(usdt_amt))
            plan.append({"type": "LIMIT", "price": p, "qty": q, "usdt": m})

        fee_buf = 1.002  # 수수료/슬리피지 버퍼(0.2%)
        required_total = sum(x["usdt"] for x in plan) * fee_buf
        return required_total, plan

    def _refresh_position(self):
        """스티키 평균가: qty>0인데 avg=0이면 이전 avg 유지."""
        old_avg = float(self.state.position_avg_price or 0.0)
        avg, qty = self.client.position_info(self.cfg.symbol, self.cfg.side)
        if qty > 0 and (avg is None or avg <= 0) and old_avg > 0:
            avg = old_avg
        self.state.position_avg_price = avg
        self.state.position_qty = qty

    def _cancel_tracked_limits(self):
        for oid in list(self.state.open_limit_ids):
            try:
                self.client.cancel_order(self.cfg.symbol, oid)
            except Exception as e:
                log(f"⚠️ 리밋 취소 실패: {oid} {e}")
        self.state.open_limit_ids.clear()

    def _ensure_tp_for_current_position(self, pp:int, step:float, min_qty:float, side:str):
        """
        현재 보유 포지션(평단/수량)을 기준으로 TP 주문을 확보/재설정한다.
        - DCA/시장가 진입은 하지 않음
        - 기존 TP가 살아있고 변화가 미미하면 유지(데드밴드)
        """
        tick = 10 ** (-pp) if pp > 0 else 0.01
        min_allowed = max(float(min_qty or 0.0), float(step or 0.0), tick)

        qty_now = float(self.state.position_qty or 0.0)
        if qty_now < min_allowed:
            return

        # entry: 평균가(0이면 1회 한정 mark/last로 대체)
        entry = float(self.state.position_avg_price or 0.0)
        if entry <= 0:
            try:
                entry = float(self.client.get_mark_price(self.cfg.symbol))
            except Exception:
                entry = float(self.client.get_last_price(self.cfg.symbol))

        # 레버리지 기준 ROI → 목표 TP 가격(틱 방향 포함)
        tp_price = tp_price_from_roi(entry, side, float(self.cfg.tp_percent), int(self.cfg.leverage), pp)

        # 수량 보정
        tp_qty = floor_to_step(qty_now, float(step or 1.0))
        if tp_qty < min_allowed:
            tp_qty = min_allowed

        if not (tp_price and tp_price > 0):
            return

        # 기존 TP 살아있고 변화가 미미하면 유지(가격 2틱, 수량 1스텝 데드밴드)
        price_changed = True
        qty_changed   = True
        if self.state.tp_order_id:
            try:
                oo = self.client.open_orders(self.cfg.symbol)
                want = str(self.state.tp_order_id)
                alive = any(str(o.get("orderId") or o.get("orderID") or o.get("id") or "") == want for o in oo)
            except Exception:
                alive = False

            if alive:
                last_price = self._last_tp_price
                last_qty   = self._last_tp_qty
                if (last_price is not None) and (abs(tp_price - last_price) < 2 * tick):
                    price_changed = False
                if (last_qty is not None) and (abs(tp_qty - last_qty)   < float(step or 1.0)):
                    qty_changed = False
                if not (price_changed or qty_changed):
                    return
                else:
                    # 변경 필요 → 기존 TP 취소
                    try:
                        self.client.cancel_order(self.cfg.symbol, self.state.tp_order_id)
                    except Exception:
                        pass
                    self.state.tp_order_id = None

        # 새 TP 배치
        tp_side = "SELL" if side.upper() == "BUY" else "BUY"
        tp_pos  = "LONG" if side.upper() == "BUY" else "SHORT"
        new_id = self.client.place_limit(
            self.cfg.symbol, tp_side, tp_qty, tp_price,
            reduce_only=False, position_side=tp_pos
        )
        self.state.tp_order_id = str(new_id)
        # 모니터링 기준값 저장
        self._last_tp_price = tp_price
        self._last_tp_qty   = tp_qty
        log(f"🎯 (attach) TP 확보: id={new_id}, price={tp_price}, qty={tp_qty}, side={tp_side}/{tp_pos}")

    # ---------- main loop ----------
    def _run(self):
        try:
            while not self._stop:
                # 0) 가용 USDT 체크
                try:
                    av = float(self.client.get_available_usdt())
                except Exception as e:
                    log(f"❌ 가용잔고 조회 실패: {e}")
                    av = 0.0
                budget = sum(float(usdt) for _, usdt in self.cfg.dca_config)
                if av < 0.99:
                    log("⛔ 가용 USDT 없음 → 종료")
                    break
                if av < budget:
                    log(f"⚠️ 가용 {av} < 계획 {budget} USDT (일부만 체결될 수 있음)")

                # 1) 정밀도/스펙 동기화
                try:
                    pp, qp = self.client.get_symbol_filters(self.cfg.symbol)
                    self.cfg.price_precision = pp
                    self.cfg.qty_precision = qp
                    log(f"ℹ️ precision synced: price={pp}, qty={qp}")
                except Exception as e:
                    log(f"⚠️ precision sync failed: {e}")
                    pp, qp = 4, 0

                spec = self.client.get_contract_spec(self.cfg.symbol)
                pp = int(spec.get("pricePrecision", pp))
                qp = int(spec.get("quantityPrecision", qp))
                contract = float(spec.get("contractSize", 1.0)) or 1.0
                min_qty = float(spec.get("minQty", 0.0))
                step = float(spec.get("qtyStep") or (1.0 if qp == 0 else 10 ** (-qp)))
                if step <= 0:
                    step = 1.0 if qp == 0 else 10 ** (-qp)
                log(f"ℹ️ spec: contractSize={contract}, minQty={min_qty}, qtyStep={step}, pp={pp}, qp={qp}")

                side = self.cfg.side.upper()
                mark = float(self.client.get_mark_price(self.cfg.symbol))

                # 1.6) === 사전 예산 점검 ===
                required, plan = self._estimate_required_margin(side, mark, spec, pp, step)
                try:
                    av = float(self.client.get_available_usdt())
                except Exception:
                    pass
                self.state.budget_ok = av + 1e-9 >= required
                self.state.budget_required = required
                self.state.budget_available = av

                if av + 1e-9 < required:
                    gap = required - av
                    log("⛔ 예산 부족: 모든 진입 주문에 필요한 증거금이 가용 USDT보다 큽니다.")
                    log(f"   필요≈{required:.4f} USDT, 가용≈{av:.4f} USDT, 부족≈{gap:.4f} USDT")
                    for idx, x in enumerate(plan, start=1):
                        log(f"   · {idx:02d} {x['type']}: price={x['price']} qty={x['qty']} → 증거금≈{x['usdt']:.4f} USDT")
                    break
                else:
                    log(f"💰 예산 확인 OK: 필요≈{required:.4f} USDT ≤ 가용≈{av:.4f} USDT")

                # === 레버리지 검증 (자동조정 없음)
                lev_now = self.client.get_current_leverage(self.cfg.symbol, self.cfg.side)
                if lev_now is not None:
                    want = float(self.cfg.leverage)
                    diff = abs(lev_now - want) / max(want, 1.0)
                    if diff > 0.02:
                        log(f"⛔ 레버리지 불일치: 설정={want}x, 거래소={lev_now}x → 수량/증거금 오차 발생")
                        log("   거래소 앱/웹에서 해당 심볼의 레버리지를 설정값과 동일하게 맞춘 뒤 다시 시작하세요.")
                        break
                else:
                    log("ℹ️ 현재 포지션이 없어 레버리지 조회값 없음 → 주문 후 다시 검증 예정")

                # ---- 사이클 시작: 기존 포지션 연결 모드 감지 ----
                try:
                    pre_avg, pre_qty = self.client.position_info(self.cfg.symbol, self.cfg.side)
                except Exception:
                    pre_avg, pre_qty = 0.0, 0.0

                min_live_qty = max(float(min_qty or 0.0), float(step or 0.0))
                attach_mode = (float(pre_qty) >= (min_live_qty * ZERO_EPS_FACTOR))

                # === attach 모드이면: 시장가+DCA 전부 스킵, TP만 확보 ===
                if attach_mode:
                    log(f"🔗 기존 포지션 연결 모드: qty={pre_qty}, avg={pre_avg} → DCA/시장가 스킵, TP 확보")
                    self.state.position_avg_price = pre_avg
                    self.state.position_qty = pre_qty
                    self._ensure_tp_for_current_position(int(pp), float(step), float(min_qty), side)

                    # 모니터링용 기준값 초기화(있으면 재사용)
                    last_entry     = float(pre_avg or 0.0)
                    last_tp_price  = self._last_tp_price
                    last_tp_qty    = self._last_tp_qty

                else:
                    # 2) 1차 시장가 진입
                    first_usdt = float(self.cfg.dca_config[0][1])
                    target_notional = first_usdt * float(self.cfg.leverage)
                    raw_qty = target_notional / max(mark * contract, 1e-12)
                    qty = floor_to_step(raw_qty, step)
                    if qty < (min_qty or step):
                        log(f"⚠️ 1차 수량이 최소수량 미달(raw={raw_qty}) → {max(min_qty, step)}로 보정")
                        qty = max(min_qty, step)

                    oid = self.client.place_market(self.cfg.symbol, side, qty)
                    if not oid:
                        raise RuntimeError("market order failed: no orderId")
                    log(f"🚀 1차 시장가 진입 주문: {oid} (투입≈{first_usdt} USDT, qty={qty})")

                    # 3) 나머지 DCA 리밋 깔기
                    cumulative = 0.0
                    self.state.open_limit_ids.clear()
                    for i, (gap_pct, usdt_amt) in enumerate(self.cfg.dca_config[1:], start=2):
                        cumulative += float(gap_pct)
                        price = mark * (1 - cumulative/100.0) if side=="BUY" else mark * (1 + cumulative/100.0)
                        price = float(f"{price:.{pp}f}")
                        target_notional = float(usdt_amt) * float(self.cfg.leverage)
                        raw_qty = target_notional / max(price * contract, 1e-12)
                        q = floor_to_step(raw_qty, step)
                        if q < (min_qty or step):
                            log(f"⚠️ {i}차 수량이 최소수량 미달(raw={raw_qty}) → {max(min_qty, step)}로 보정")
                            q = max(min_qty, step)
                        lid = self.client.place_limit(self.cfg.symbol, side, q, price)
                        self.state.open_limit_ids.append(str(lid))
                        log(f"🧩 {i}차 리밋: id={lid}, price={price}, qty={q}, 투입≈{usdt_amt}USDT")

                    # 4) 초기 TP 세팅
                    self._refresh_position()
                    tick = 10 ** (-pp) if pp > 0 else 0.01
                    min_allowed = max(float(min_qty or 0.0), float(step or 0.0), tick)
                    qty_now = float(self.state.position_qty or 0.0)

                    last_entry = None
                    last_tp_price = None
                    last_tp_qty = None

                    if qty_now >= min_allowed:
                        entry = float(self.state.position_avg_price or 0.0)
                        if entry <= 0:
                            try:
                                entry = float(self.client.get_mark_price(self.cfg.symbol))
                            except Exception:
                                entry = float(self.client.get_last_price(self.cfg.symbol))
                            log(f"⚠️ avg_price=0 → fallback entry={entry} (initial only)")

                        tp_price = tp_price_from_roi(entry, side, float(self.cfg.tp_percent), int(self.cfg.leverage), pp)

                        tp_qty = floor_to_step(qty_now, float(step or 1.0))
                        if tp_qty < min_allowed:
                            tp_qty = min_allowed
                        if tp_price <= 0 or tp_qty <= 0:
                            raise RuntimeError(f"TP invalid: price={tp_price}, qty={tp_qty}")

                        tp_side = "SELL" if side == "BUY" else "BUY"
                        tp_pos  = "LONG" if side == "BUY" else "SHORT"
                        new_tp_id = self.client.place_limit(
                            self.cfg.symbol,
                            tp_side,
                            tp_qty,
                            tp_price,
                            reduce_only=False,                 # HEDGE에선 reduceOnly 쓰지 않음
                            position_side=tp_pos_side,         # "LONG" / "SHORT"
                            close_position=True                # ✅ 반드시 닫기 전용
                        )

                        self.state.tp_order_id = str(new_tp_id)
                        last_entry = entry
                        last_tp_price = tp_price
                        last_tp_qty = tp_qty
                        self._last_tp_price = tp_price
                        self._last_tp_qty   = tp_qty
                        log(f"🎯 TP 배치 완료: id={new_tp_id}, price={tp_price}, qty={tp_qty}, side={tp_side}/{tp_pos}")
                    else:
                        log("ℹ️ 포지션 없음 또는 최소단위 미만 → TP 생략")
                        last_entry = None
                        last_tp_price = None
                        last_tp_qty = None

                # ===== 5) 모니터링 루프 =====
                tp_reset_cooldown = 3.0
                last_tp_reset_ts = 0.0
                zero_streak = 0  # 종료 판정 연속 횟수

                while not self._stop:
                    time.sleep(POLL_SEC)
                    self._refresh_position()

                    # 오픈오더 조회(한번만)
                    try:
                        open_orders = self.client.open_orders(self.cfg.symbol)
                    except Exception as e:
                        log(f"⚠️ 오픈오더 조회 실패: {e}")
                        open_orders = []

                    # TP 생존 확인
                    tp_alive = False
                    if self.state.tp_order_id:
                        want = str(self.state.tp_order_id)
                        for o in open_orders:
                            oid = str(o.get("orderId") or o.get("orderID") or o.get("id") or "")
                            if oid == want:
                                tp_alive = True
                                break

                    # ----- 종료 판정 (연속 N회 + TP 미생존 + 이중확인) -----
                    tick = 10 ** (-pp) if pp > 0 else 0.01
                    min_allowed = max(float(min_qty or 0.0), float(step or 0.0), tick)
                    zero_eps = min_allowed * ZERO_EPS_FACTOR

                    qty_now = float(self.state.position_qty or 0.0)
                    if qty_now < zero_eps:
                        zero_streak += 1
                    else:
                        zero_streak = 0

                    really_closed = (zero_streak >= CLOSE_ZERO_STREAK) and (not tp_alive)
                    if really_closed:
                        # 이중확인
                        try:
                            chk_avg, chk_qty = self.client.position_info(self.cfg.symbol, self.cfg.side)
                        except Exception:
                            chk_avg, chk_qty = 0.0, 0.0
                        if float(chk_qty or 0.0) < zero_eps:
                            # 정리
                            self._cancel_tracked_limits()
                            if self.state.tp_order_id:
                                try:
                                    self.client.cancel_order(self.cfg.symbol, self.state.tp_order_id)
                                except Exception:
                                    pass
                                self.state.tp_order_id = None
                            log("✅ 포지션 종료 확정(연속검증+이중확인) → 대기")
                            break
                        else:
                            zero_streak = 0  # 오탐이었다면 초기화

                    # ----- TP 재설정(데드밴드 + 쿨다운) -----
                    need_reset_tp = False
                    entry_now = float(self.state.position_avg_price or 0.0)
                    if entry_now <= 0 and last_entry and last_entry > 0:
                        entry_now = last_entry

                    if not tp_alive:
                        need_reset_tp = (qty_now >= min_allowed and entry_now > 0)
                    else:
                        if qty_now >= min_allowed and entry_now > 0:
                            # 이상적 TP
                            ideal_price = tp_price_from_roi(entry_now, side, float(self.cfg.tp_percent), int(self.cfg.leverage), pp)
                            ideal_qty   = max(floor_to_step(qty_now, step), min_allowed)
                            # 데드밴드
                            if (last_entry is None) or (last_tp_price is None) or (last_tp_qty is None):
                                need_reset_tp = True
                            elif (abs(entry_now - last_entry) >= 2 * tick) or \
                                 (abs(ideal_price - last_tp_price) >= 2 * tick) or \
                                 (abs(ideal_qty - last_tp_qty) >= step):
                                need_reset_tp = True

                    if need_reset_tp:
                        now_ts = self._now()
                        if now_ts - last_tp_reset_ts < tp_reset_cooldown:
                            continue

                        # 기존 TP 살아있으면 취소 → 반영 대기
                        if self.state.tp_order_id and tp_alive:
                            try:
                                self.client.cancel_order(self.cfg.symbol, self.state.tp_order_id)
                                self._wait_cancel(self.state.tp_order_id, timeout=2.5)
                            except Exception as e:
                                log(f"⚠️ TP 취소 실패(무시): {e}")

                        if entry_now <= 0 or qty_now < min_allowed:
                            continue

                        new_price = tp_price_from_roi(entry_now, side, float(self.cfg.tp_percent), int(self.cfg.leverage), pp)
                        new_qty   = max(floor_to_step(qty_now, step), min_allowed)
                        new_side  = "SELL" if side == "BUY" else "BUY"
                        new_pos   = "LONG" if side == "BUY" else "SHORT"

                        try:
                            new_id = self.client.place_limit(
                                self.cfg.symbol, new_side, new_qty, new_price,
                                reduce_only=False, position_side=new_pos
                            )
                        except Exception as e:
                            # 80001(가용 부족)은 보류
                            if "80001" in str(e):
                                continue
                            else:
                                raise

                        self.state.tp_order_id = str(new_id)
                        last_entry = entry_now
                        last_tp_price = new_price
                        last_tp_qty = new_qty
                        self._last_tp_price = new_price
                        self._last_tp_qty   = new_qty
                        last_tp_reset_ts = now_ts
                        log(f"♻️ TP 재설정: id={new_id}, price={new_price}, qty={new_qty}")

                # 루프 탈출: repeat면 다시 반복
                if self._stop:
                    break

                if not self.state.repeat_mode:
                    break
                else:
                    # 남은 오더 안전정리
                    self._cancel_tracked_limits()
                    if self.state.tp_order_id:
                        try:
                            self.client.cancel_order(self.cfg.symbol, self.state.tp_order_id)
                        except:
                            pass
                        self.state.tp_order_id = None
                    self.state.reset_orders()

                    # 쿨다운(기본 60초). 정지 신호 들어오면 즉시 중단
                    delay = max(0, RESTART_DELAY_SEC)
                    if delay > 0:
                        log(f"🔁 반복 모드 → {delay}초 대기 후 재시작")
                        for _ in range(delay):
                            if self._stop:
                                break
                            time.sleep(1)

                    if self._stop:
                        break

                    log("🔁 재시작")
                    continue

        except Exception as e:
            log(f"❌ 런타임 오류: {e}")
        finally:
            self.state.running = False
            log("⏹️ 봇 종료")
