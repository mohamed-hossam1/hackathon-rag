import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configures application-wide logging format and handlers."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


# Automatically configure default logging when src package is imported
setup_logging()

logger = logging.getLogger("medical_rag")
