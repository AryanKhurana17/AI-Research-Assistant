"""
Logging configuration for the AI Research Assistant.

Provides structured logging for:
- Query routing decisions
- Tool calls and results
- RAG retrieval details
- Agent execution times

Usage:
    from src.logging_config import setup_logging
    setup_logging()  # Call once at application startup

    # Then in any module:
    import logging
    logger = logging.getLogger("research_assistant.tools.calculator")
    logger.info("Calculator called with: %s", expression)
"""
import logging
import sys
from pathlib import Path


LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Application-wide logger name prefix
APP_LOGGER_NAME = "research_assistant"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the application.

    Sets up both console and file handlers with structured formatting.
    The file handler always logs at DEBUG level for full traceability.

    Args:
        level: Console log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        The root application logger.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-35s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler (always DEBUG for full audit trail)
    file_handler = logging.FileHandler(LOG_DIR / "research_assistant.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Root application logger
    root_logger = logging.getLogger(APP_LOGGER_NAME)
    root_logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        module_name: Dotted name for the module (e.g., "tools.calculator").

    Returns:
        A logger instance under the application namespace.
    """
    return logging.getLogger(f"{APP_LOGGER_NAME}.{module_name}")
