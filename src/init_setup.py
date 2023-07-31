import os
import configparser
import logging


class Setup:
    """
    Set root path, config path, and logging options

    Logging levels:
    logging hierarchy (each above contains every below):
    logging.NOTSET 0
    logging.DEBUG 10
    logging.INFO 20
    logging.WARNING 30
    logging.ERROR 40
    logging.CRITICAL 50
    """
    def __init__(self) -> None:

        # Set the root dir
        abs_path = os.path.abspath(__file__)
        dir_name = os.path.dirname(abs_path)
        os.chdir(dir_name)
        os.chdir('..')
        self.ROOT_PATH = os.getcwd().replace("\\", "/")+"/"

        # Specify config path
        self.CONFIG_PATH = self.ROOT_PATH+"/config.ini"
        self.config = configparser.ConfigParser()
        self.config.read(self.CONFIG_PATH)

        logging.basicConfig(
            # filename="../reports/logs/logger_"+time.strftime("%Y%m%d-%H%M%S")+".log",
            level=int(self.config["logger"]["LoggerLevel"]),
            format="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger("Config")
        self.logger.info(f"Root Path: {self.ROOT_PATH}")
