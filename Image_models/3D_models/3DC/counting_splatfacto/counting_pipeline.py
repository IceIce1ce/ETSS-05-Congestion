from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import torch
import os
import torchvision.utils as vutils
import numpy as np

@dataclass
class CountingPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: CountingPipeline)

class CountingPipeline(VanillaPipeline):
    config: CountingPipelineConfig

    def __init__(self, config: CountingPipelineConfig, **kwargs):
        super().__init__(config, **kwargs)

    def get_average_eval_image_metrics(self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False):
        metrics_dict = super().get_average_eval_image_metrics()
        self.eval()
        idx = 0
        for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
            outputs = self.model.get_outputs_for_camera(camera=camera)
            metrics_dict, depth_dict = self.get_depth_metrics_and_images(outputs, batch)
            if output_path is not None:
                os.makedirs(os.path.join(output_path, "acc"), exist_ok=True)
                os.makedirs(os.path.join(output_path, "raw_depth"), exist_ok=True)
                print("Saving depth + acc on image", idx)
                key = "raw_depth"
                raw_depth = depth_dict[key].squeeze().cpu().numpy()
                np.save(output_path / f"raw_depth/raw_depth_{idx:04d}", raw_depth)
                vutils.save_image(outputs["accumulation"].squeeze().unsqueeze(0).cpu(), output_path / f"acc/acc_{idx:04d}.png")
                np.save(output_path / f"acc/acc_{idx:04d}", outputs["accumulation"].squeeze().cpu().numpy())
            idx += 1
        return metrics_dict

    def get_depth_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_depth = batch.get("depth", None)
        predicted_depth = outputs["depth"]
        min_depth = predicted_depth.min()
        max_depth = predicted_depth.max()
        normalized_depth = (predicted_depth - min_depth) / (max_depth - min_depth + 1e-8)
        predicted_depth = predicted_depth[None, None, :, :]
        if gt_depth is not None:
            gt_depth = gt_depth[None, None, :, :]
        metrics_dict = {}
        if gt_depth is not None:
            abs_rel = torch.mean(torch.abs(predicted_depth - gt_depth) / (gt_depth + 1e-8))
            rmse = torch.sqrt(torch.mean((predicted_depth - gt_depth) ** 2))
            metrics_dict["abs_rel"] = float(abs_rel.item())
            metrics_dict["rmse"] = float(rmse.item())
        depth_image = ((normalized_depth * 255).clamp(0, 255).byte())
        depth_dict = {"depth": depth_image, "raw_depth": predicted_depth}
        return metrics_dict, depth_dict