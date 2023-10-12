import logging
import logging.handlers
import time
import os
import sys
from typing import Optional


FOMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)8s]: %(message)s',  
    "%Y-%m-%d %H:%M:%S"
)


def init_logger(logger_name, log_dir, log_file_name: Optional[str]=None):
    if log_file_name is None:
        log_file_name = time.strftime(str(log_dir / "%Y-%m-%d_%H-%M-%S")) + ".log"
    else:
        log_file_name = str(log_dir / log_file_name.replace(".log", "")) + ".log"

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler and set level to debug
    handler_logfile = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=1024*1024, backupCount=10)
    handler_logfile.setLevel(logging.DEBUG)
    handler_logfile.setFormatter(FOMATTER)
    logger.addHandler(handler_logfile)

    # create terminal handler and set level to info '''
    handler_terminal = logging.StreamHandler(stream=sys.stdout)
    handler_terminal.setLevel(logging.DEBUG)
    handler_terminal.setFormatter(FOMATTER)
    logger.addHandler(handler_terminal)

    return logger