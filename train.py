# implementation of Pi-Flow for simple minded people like @cloneofsimo or like me.
# inspired and heavily based on: 
# Cloneofsimo minRF: https://github.com/cloneofsimo/minRF/tree/main
# Flow official: https://github.com/Lakonik/Flow/tree/main

import argparse
import torch
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(Path(folder_path).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # No labels for unconditional
        return img, 0


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

if __name__ == "__main__":
    # train class conditional RF on mnist.
    import os
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    try:
        import wandb
        is_wandb_available = True
    except ImportError:
        is_wandb_available = False

    from dit import DiT_Llama

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--attn", type=str, default="favor")
    parser.add_argument("--iter", type=int, default=2, help="number of analytical integration per training step")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    result_dir = f"contents/{args.dataset}/epoch_{args.epochs}-batch-size_{args.batch_size}-attn_{args.attn}-iter_{args.iter}"
    weight_dir = f"weights/{args.dataset}/epoch_{args.epochs}-batch-size_{args.batch_size}-attn_{args.attn}-iter_{args.iter}"
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # for reproducibility
    import sys
    with open(sys.argv[0]) as f:
        code = f.read()
    with open(f"{result_dir}/flow.txt", "w") as f:
        f.write(code)

    if args.dataset == "cifar":
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        patch_size = 2
        model = DiT_Llama(
            channels, 32, dim=128, n_layers=10, n_heads=8, num_classes=10, patch_size=patch_size
        ).cuda()

    elif args.dataset == "mnist":
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        patch_size = 2
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, patch_size=patch_size
        ).cuda()

    elif args.dataset == "celeba_hq":
        data_path = "data/celeba_hq_256"  # path to your flat folder
        transform = transforms.Compose([
            transforms.Resize(256),          # keep original 256x256 or downsample
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        fdatasets = FlatImageDataset(data_path, transform=transform)
        channels = 3
        patch_size = 4  
        num_classes = 1     
        model = DiT_Llama(
            channels, 256, dim=128, n_layers=12, n_heads=8, num_classes=num_classes, patch_size=patch_size
        ).cuda()


    rf = Flow(model, iter=args.iter)
    if args.dataset in ["mnist", "cifar"]:
        train_ds = fdatasets(root="./data", train=True, download=True, transform=transform)
    else:
        train_ds = fdatasets
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if is_wandb_available:
        wandb.init(project=f"flow", name=f"{args.dataset}_epoch_{args.epochs}-batch-size_{args.batch_size}_attn_{args.attn}-iter_{args.iter}")

    # train model (if there is no model checkpoint)
    optimizer = optim.Adam(rf.model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler()  # optional for bf16
    for epoch in range(args.epochs):
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, c in train_dl:
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = rf.forward_fm(x, c)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            if is_wandb_available:
                wandb.log({"loss": loss.item()})
    torch.save(rf.model.state_dict(), f"{weight_dir}/model.pth")

    # Determine input shape and conditioning based on dataset
    if args.dataset in ["mnist", "cifar"]:
        H = W = 32
        B = 16
        cond = torch.arange(0, B).cuda() % 10  # class labels
        channels = 1 if args.dataset == "mnist" else 3
    elif args.dataset == "celeba_hq":
        H = W = 256
        B = 16
        channels = 3
        # Use attributes if model is conditional; else zeros for unconditional
        if model.num_classes > 1:  # assume num_classes=40 for CelebA-HQ attributes
            cond = torch.zeros(B, model.num_classes, dtype=torch.float).cuda()  # replace with attribute tensor if available
        else:
            cond = torch.zeros(B, dtype=torch.long).cuda()  # unconditional
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for sampling")

    # Initialize noise
    x_T = torch.randn(B, channels, H, W, generator=torch.Generator().manual_seed(42)).cuda()

    # Sample from model
    images = rf.sample_fm(x_T, cond, DDIM_NFE=50)

    # Save to GIF / PNG
    gif = tensors_to_gif(images)
    gif[0].save(f"{result_dir}/sample_fm.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
    gif[-1].save(f"{result_dir}/sample_fm_last.png")

