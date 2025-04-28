import torch
from models import vgg19
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def predict(inp):
    inp = Image.open(inp).convert('RGB')
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    inp = inp.cuda()
    with torch.set_grad_enabled(False):
        outputs, _ = model(inp)
    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)

if __name__ == '__main__':
    # model
    model_path = "checkpoints/model_qnrf.pth"
    model = vgg19()
    model.cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    title = "Distribution Matching for Crowd Counting"
    desc = "A demo of DM-Count, a NeurIPS 2020 paper by Wang et al. Outperforms the state-of-the-art methods by a " \
           "large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50. " \
           "This demo uses the QNRF trained model. Try it by uploading an image or clicking on an example " \
           "(could take up to 20s if running on CPU)."
    examples = [["assets/3.png"], ["assets/2.png"], ["assets/1.png"]]
    inputs = gr.inputs.Image(label="Image of Crowd", type='filepath')
    outputs = [gr.outputs.Image(label="Predicted Density Map", type='filepath'), gr.outputs.Label(label="Predicted Count")]
    gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc, examples=examples, allow_flagging=False).launch()
