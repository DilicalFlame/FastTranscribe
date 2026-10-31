"""Start FastTranscribe:  uv run run.py"""
import threading
import webbrowser

import uvicorn

from server.config import settings

if __name__ == "__main__":
    url = f"http://{settings.host}:{settings.port}"
    print(f"\n  FastTranscribe → {url}\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("server.main:app", host=settings.host, port=settings.port, log_level="warning")
