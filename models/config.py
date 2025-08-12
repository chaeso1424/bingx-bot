from dataclasses import dataclass, field
from typing import List, Tuple

# DCA 설정: [(간격% (누적형), 투자 USDT), ...]
# 예) [(0.0, 5), (0.4, 5), (0.8, 10)]  => 1차 시장가 5USDT, 2차 -0.4%, 3차 -1.2%
@dataclass
class BotConfig:
    symbol: str = "BTC-USDT"
    side: str = "BUY"                    # BUY(롱) / SELL(숏)
    margin_mode: str = "CROSS"           # CROSS / ISOLATED
    leverage: int = 10
    tp_percent: float = 4.5
    min_qty: float = 0.0                 # 거래소 최소수량(어댑터에서 채움)
    price_precision: int = 4
    qty_precision: int = 3
    dca_config: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 5.0), (0.4, 5.0), (0.8, 10.0)])
