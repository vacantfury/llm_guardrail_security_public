import colorlog
import logging

DEFAULT_LOGGER_NAME = "llm_guardrail_security"


def get_logger(name: str = DEFAULT_LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = colorlog.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        for h in logging.root.handlers[:]:
            logger.root.removeHandler(h)

        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(levelname)s]%(asctime)s %(name)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
