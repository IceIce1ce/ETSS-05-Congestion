from dataclasses import dataclass, field
from typing import Type
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerClick, ViewerControl
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
import numpy as np
import torch
import viser.transforms as vtf
import trimesh
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

@dataclass
class CountingSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: CountingSplatfactoModel)

class CountingSplatfactoModel(SplatfactoModel):
    config: CountingSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
        self.viewer_control = ViewerControl()

    def _click_gaussian(self, button: ViewerButton):
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.eval()
        with torch.no_grad():
            outputs = self.get_outputs(cam.to(self.device))
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()
        self.train()
        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)
        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(name=f"/click", mesh=sphere_mesh, position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location)