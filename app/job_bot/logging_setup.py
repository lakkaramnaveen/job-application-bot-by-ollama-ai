import logging
import os


def configure_logging() -> None:
    """Configure root logging once, at process start. Level comes from the
    LOG_LEVEL env var (default INFO) so it doesn't need its own .env key.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Playwright's own logger is noisy at INFO; keep it quiet unless debugging.
    logging.getLogger("playwright").setLevel(logging.WARNING)
    # Same for httpx/httpcore: at INFO, httpx logs one "HTTP Request: POST
    # .../api/chat ..." line per LLM call - several per posting - which
    # buried the lines a run actually prints for the user ("Applied: ...",
    # errors). Real failures still surface as exceptions/printed errors;
    # LOG_LEVEL=DEBUG brings these back for debugging.
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
