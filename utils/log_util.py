from loguru import logger
import logging

def get_logger():
    logger.add("app.log", rotation="500 MB", compression="zip", backtrace=True, diagnose=True)
    return logger

