import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):
    def __init__(self, args, cost_class: float = 1, cost_point: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.args = args
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_point = outputs["pred_points"].flatten(0, 1)
        tgt_point = torch.cat([v["points"] for v in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_point = torch.cdist(out_point, tgt_point.cuda(), p=1)
        C = self.cost_class * cost_class + self.cost_point * cost_point
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["points"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(args, cost_class=args.set_cost_class, cost_point=args.set_cost_point, cost_giou=args.set_cost_giou)