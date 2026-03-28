from .counter import Counter

def build_model(model_config):
    return Counter(model_config.NAME)