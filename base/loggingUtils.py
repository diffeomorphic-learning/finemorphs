import logging
import os


def setup_default_logging(output_dir=None, config=None, fileName=None, stdOutput=True, mode='w'):
    logger = logging.getLogger()
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    if config is None:
        log_file_name = fileName
    else:
        log_file_name = config.log_file_name
    if output_dir is None:
        output_dir = ""
    if fileName is not None:
        if not os.access(output_dir, os.W_OK):
            if os.access(output_dir, os.F_OK):
                logging.error('Cannot save in ' + output_dir)
                return
            else:
                os.makedirs(output_dir)
        fh = logging.FileHandler("%s/%s" % (output_dir, log_file_name), mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if stdOutput:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

