from threading import Lock

class BotState:
    def __init__(self):
        self.running = False
        self.repeat_mode = False
        self.position_avg_price = 0.0
        self.position_qty = 0.0
        self.tp_order_id = None
        self.open_limit_ids = []
        self._lock = Lock()

    def reset_orders(self):
        with self._lock:
            self.tp_order_id = None
            self.open_limit_ids = []
