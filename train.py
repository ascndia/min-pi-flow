# implementation of Pi-Flow for simple minded people like @cloneofsimo or like me.
# inspired and heavily based on: 
# Cloneofsimo minRF: https://github.com/cloneofsimo/minRF/tree/main
# Flow official: https://github.com/Lakonik/Flow/tree/main

import argparse
import torch
import torch.nn.functional as F

class Flow:
    def __init__(self, model, DDIM_NFE=128, iter=2):
        self.model = model
        self.iter = iter
        self.DDIM_NFE = DDIM_NFE

        self.in_channels = student_model.in_channels
        self.input_size = student_model.input_size

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
    parser.add_argument("--K", type=int, default=8, help="number of Gaussian mixture components")
    parser.add_argument("--iter", type=int, default=2, help="number of analytical integration per training step")
    args = parser.parse_args()

    result_dir = f"contents/{args.dataset}/K_{args.K}-iter_{args.iter}"
    weight_dir = f"weights/{args.dataset}/K_{args.K}-iter_{args.iter}"
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
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, patch_size=patch_size
        ).cuda()
        student_model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, K=args.K, patch_size=patch_size
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
        student_model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, K=args.K, patch_size=patch_size
        ).cuda()

    rf = Flow(model, iter=args.iter)
    train_ds = fdatasets(root="./data", train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)

    if is_wandb_available:
        wandb.init(project=f"flow", name=f"{args.dataset}-K_{args.K}-iter_{args.iter}")

    # train model (if there is no model checkpoint)
    epochs = 25
    optimizer = optim.Adam(rf.model.parameters(), lr=5e-4)
    for epoch in tqdm(range(epochs), desc="Training model"):
        for i, (x, c) in enumerate(train_dl):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss = rf.forward_fm(x, c)
            loss.backward()
            optimizer.step()
            
            if is_wandb_available:
                wandb.log({"loss": loss.item()})
    torch.save(rf.model.state_dict(), f"{weight_dir}/model.pth")

    # Generate samples from model with NFE = 50
    x_T = torch.randn(16, channels, 32, 32, generator=torch.Generator().manual_seed(42)).cuda()
    cond = torch.arange(0, 16).cuda() % 10
    images = rf.sample_fm(x_T, cond, DDIM_NFE=50)
    gif = tensors_to_gif(images)
    gif[0].save(f"{result_dir}/sample_fm.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
    gif[-1].save(f"{result_dir}/sample_fm_last.png")

