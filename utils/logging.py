# utils/logging.py
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# 로그 파일 경로
LOG_FILE = Path("./logs/logs.txt")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# 핸들러 설정: 자정마다 새로운 로그 파일, 7일치만 보관
handler = TimedRotatingFileHandler(
    LOG_FILE,
    when="midnight",   # 자정 기준 회전
    interval=1,        # 1일 단위
    backupCount=7,     # 최근 7개 파일만 보관
    encoding="utf-8"
)

# 로그 포맷
formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)

# 로거 생성
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def log(msg: str) -> None:
    """로그 출력 + 파일 저장 (자동 날짜 회전)"""
    logger.info(msg)
    print(f"[{logging.Formatter.formatTime(formatter, None)}] {msg}", flush=True)
