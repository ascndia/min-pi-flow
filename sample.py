# sample.py
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from pathlib import Path

from dit import DiT_Llama

class Flow:
    def __init__(self, model, DDIM_NFE=128, iter=2):
        self.model = model
        self.iter = iter
        self.DDIM_NFE = DDIM_NFE

        self.in_channels = model.in_channels
        self.input_size = model.input_size

    # flow matching loss
    def forward_fm(self, z0, cond):
        b = z0.size(0)
        nt = torch.randn((b,)).to(z0.device)
        t = torch.sigmoid(nt)
        texp = t.view([b, *([1] * len(z0.shape[1:]))])
        z1 = torch.randn_like(z0)
        zt = (1 - texp) * z0 + texp * z1
        vtheta = self.model(zt, t, cond)
        loss = F.mse_loss(vtheta, z1 - z0)
        return loss.mean()

    # flow matching sampling
    @torch.no_grad()
    def sample_fm(self, x_t, cond, DDIM_NFE=128):
        b = x_t.size(0)
        dt = 1.0 / DDIM_NFE
        dt = torch.tensor([dt] * b).to(x_t.device).view([b, *([1] * len(x_t.shape[1:]))])
        images = [x_t]
        for i in range(DDIM_NFE, 0, -1):
            t = i / DDIM_NFE
            t = torch.tensor([t] * b).to(x_t.device)
            v_t = self.model(x_t, t, cond)
            x_t = x_t - dt * v_t
            images.append(x_t)
        return images

# helper function to convert list of tensors to gif
def tensors_to_gif(images):
    """Convert list of tensors to gif"""
    gif = []
    for image in images:
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        x_as_image = make_grid(image.float(), nrow=4)
        img = x_as_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gif.append(Image.fromarray(img))
    return gif

@torch.no_grad()
def main(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Configuration (must match training) ----
    channels = 3
    image_size = 256
    patch_size = 4
    dim = 128
    n_layers = 12
    n_heads = 8
    num_classes = 1      # unconditional CelebA-HQ
    ddim_nfe = 50
    batch = 16
    result_dir = Path("samples")
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---- Recreate model + Flow ----
    model = DiT_Llama(
        channels, image_size, dim=dim,
        n_layers=n_layers, n_heads=n_heads,
        num_classes=num_classes, patch_size=patch_size
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rf = Flow(model, iter=2)

    # ---- Prepare noise + condition ----
    x_T = torch.randn(batch, channels, image_size, image_size, generator=torch.Generator().manual_seed(42)).to(device)
    cond = torch.zeros(batch, dtype=torch.long, device=device)

    # ---- Sampling ----
    print("Sampling...")
    images = rf.sample_fm(x_T, cond, DDIM_NFE=ddim_nfe)

    # ---- Save GIF ----
    gif = tensors_to_gif(images)
    gif_path = result_dir / "sample_fm.gif"
    gif[0].save(gif_path, save_all=True, append_images=gif[1:], duration=100, loop=0)
    gif[-1].save(result_dir / "sample_fm_last.png")
    print(f"Saved to {gif_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args.model_path)
