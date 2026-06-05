"""
config/logging_config.py
Configuração centralizada de logging com structlog.
"""

from __future__ import annotations

import sys
import logging
import structlog

from config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configura structlog e logging padrão do Python."""

    log_level = getattr(logging, settings.log_level, logging.INFO)

    # Processors comuns
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [console_handler]

    # File handler (se configurado)
    if settings.log_file:
        try:
            file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except OSError:
            pass  # Não bloqueia se não conseguir escrever no arquivo

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Silencia loggers verbosos de bibliotecas
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Retorna um logger structlog nomeado."""
    return structlog.get_logger(name)
