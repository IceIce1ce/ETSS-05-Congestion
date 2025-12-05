import re
import torch

def interpolate_pos_embed_multimae(model, checkpoint_model):
    pattern = "[in]*put_adapters\.(.*)\.pos_emb"
    matched_keys = [k for k in checkpoint_model if bool(re.match(pattern, k))]
    for key in matched_keys:
        domain = re.match(pattern, key).group(1)
        if getattr(model.input_adapters, domain, None) is not None:
            pos_embed_checkpoint = checkpoint_model[key]
            _, _, orig_H, orig_W = pos_embed_checkpoint.shape
            _, _, new_H, new_W = getattr(model.input_adapters, domain).pos_emb.shape
            if (orig_H != new_H) or (orig_W != new_W):
                pos_embed_checkpoint = torch.nn.functional.interpolate(pos_embed_checkpoint, size=(new_H, new_W), mode='bicubic', align_corners=False)
                checkpoint_model[key] = pos_embed_checkpoint
        if getattr(model.output_adapters, domain, None) is not None:
            pos_embed_checkpoint = checkpoint_model[key.replace('input', 'output')]
            _, _, orig_H, orig_W = pos_embed_checkpoint.shape
            _, _, new_H, new_W = getattr(model.output_adapters, domain).pos_emb.shape
            if (orig_H != new_H) or (orig_W != new_W):
                pos_embed_checkpoint = torch.nn.functional.interpolate(pos_embed_checkpoint, size=(new_H, new_W), mode='bicubic', align_corners=False)
                checkpoint_model[key.replace('input', 'output')] = pos_embed_checkpoint
    if 'fuse.pos_emb' in checkpoint_model.keys():
        pos_embed_checkpoint_fus = checkpoint_model['fuse.pos_emb']
        _, _, orig_H, orig_W = pos_embed_checkpoint_fus.shape
        _, _, new_H, new_W = model.fuse.pos_emb.shape
        if (orig_H != new_H) or (orig_W != new_W):
            pos_embed_checkpoint_fus = torch.nn.functional.interpolate(pos_embed_checkpoint_fus, size=(new_H, new_W), mode='bicubic', align_corners=False)
            checkpoint_model['fuse.pos_emb'] = pos_embed_checkpoint_fus