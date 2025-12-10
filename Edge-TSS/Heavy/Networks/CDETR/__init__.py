from .conditional_detr import build

def build_model(args):
    model, criterion, postprocessors = build(args)
    return model, criterion, postprocessors