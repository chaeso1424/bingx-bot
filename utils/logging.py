from datetime import datetime
from pathlib import Path

# Log file in the current working directory
LOG_FILE = Path("./logs.txt")

def log(msg: str) -> None:
    """Log a message with a timestamp to stdout and the log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    # Print to console immediately
    print(line, end="", flush=True)
    # Append to log file
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)