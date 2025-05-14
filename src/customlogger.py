import logging

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s :: [%(levelname)-8s] :: {%(pathname)s:%(lineno)d} :: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.CRITICAL)