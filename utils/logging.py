from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    """로그 메시지를 타임스탬프와 함께 출력하고 파일에 저장한다.

    :param msg: 기록할 메시지 문자열
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    # 콘솔에 출력
    print(line, flush=True)
    # logs.txt 파일에 덧붙여 쓰기
    log_file = Path("logs.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")