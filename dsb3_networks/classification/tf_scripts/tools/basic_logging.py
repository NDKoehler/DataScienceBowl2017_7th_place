import logging
import errno
import os

def initialize_logger(folder=''):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if (folder != ''):
        try:
            os.stat(folder)
        except:
            try:
                os.mkdir(folder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise Exception(
                        ("Could create folder: {}").format(folder))


        # create error file handler and set level to error
        handler = logging.FileHandler(os.path.join(
            folder, "error.log"), "w", encoding=None, delay="true")
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S | line:%d')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(os.path.join(folder, "all.log"), "a")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S | line:%d')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S | line:%d')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
