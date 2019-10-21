import logging

logger = None


def get_logger():
    global logger

    if not logger:
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s(%(lineno)d):%(funcName)s|: %(message)s',
                                      datefmt='%H:%M:%S')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger("root")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger
