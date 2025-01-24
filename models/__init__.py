from .lavcaps import LAVCAP

def load_model(config):
    return LAVCAP.from_config(config)