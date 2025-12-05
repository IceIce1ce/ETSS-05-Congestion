from .registry import is_model_in_modules, model_entrypoint

def split_model_name(model_name):
    model_split = model_name.split(':', 1)
    if len(model_split) == 1:
        return '', model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ('timm', 'hf_hub')
        return source_name, model_name

def create_model(model_name, **kwargs):
    source_name, model_name = split_model_name(model_name)
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'. Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate
    kwargs = {k: v for k, v in kwargs.items()}
    create_fn = model_entrypoint(model_name)
    model = create_fn(**kwargs)
    return model