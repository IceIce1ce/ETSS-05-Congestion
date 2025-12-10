import math
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import NestedTensor, nested_tensor_from_tensor_list, accuracy, get_world_size, interpolate, is_dist_avail_and_initialized, inverse_sigmoid
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss, sigmoid_focal_loss
from .transformer import build_transformer

class ConditionalDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, channel_point, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.point_embed = MLP(hidden_dim, hidden_dim, channel_point, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.point_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_points': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_points, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_points, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_points):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        loss_point = F.l1_loss(src_points, target_points, reduction='none')
        losses = {}
        losses['loss_point'] = loss_point.sum() / num_points
        return losses

    def loss_masks(self, outputs, targets, indices, num_points):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_points), "loss_dice": dice_loss(src_masks, target_masks, num_points)}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'points': self.loss_points, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_points, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_points = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        points = box_ops.box_cxcywh_to_xyxy(out_point)
        points = torch.gather(points, 1, topk_points.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        points = points * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'points': b} for s, l, b in zip(scores, labels, points)]
        return results

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args):
    num_classes = 2 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = ConditionalDETR(backbone, transformer, num_classes=num_classes, num_queries=args.num_queries, channel_point = args.channel_point, aux_loss=args.aux_loss)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_point': args.point_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'points', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)
    postprocessors = {'point': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    return model, criterion, postprocessors