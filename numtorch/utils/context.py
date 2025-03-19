from config import Config


class eval:
    def __init__(self):
        global Config
        self.old_value: bool = Config.NUMTORCH_EVAL

    def __enter__(self):
        global Config
        Config.NUMTORCH_EVAL = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global Config
        Config.NUMTORCH_EVAL = self.old_value
