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
    @torch.no_grad() # Keep no_grad for sampling
    def sample_fm(self, x_t, cond, DDIM_NFE=128):
        b = x_t.size(0)
        dt = 1.0 / DDIM_NFE
        # Ensure dt matches the expected dtype (like bfloat16) and device
        dt_tensor = torch.tensor([dt] * b, device=x_t.device, dtype=torch.bfloat16).view([b, *([1] * (x_t.dim() - 1))])

        images = [x_t.clone().cpu()] # Store initial noise (maybe on CPU if memory is tight)

        for i in tqdm(range(DDIM_NFE, 0, -1), desc="Sampling"): # Add tqdm progress bar
            t = i / DDIM_NFE
            # Create time tensor matching batch size, device, and expected dtype (bfloat16)
            t_tensor = torch.tensor([t] * b, device=x_t.device, dtype=torch.bfloat16)

            # Ensure x_t is bfloat16 before entering the model
            x_t = x_t.to(dtype=torch.bfloat16)
            # Ensure cond is also the correct dtype if it's used by the model directly
            cond = cond.to(dtype=torch.bfloat16) # Or keep as long if used for embedding lookup

            # --- FIX: Add torch.amp.autocast ---
            with autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
                # Model call happens inside BF16 autocast context
                # self.model expects BF16 inputs for its parameters now
                v_t = self.model(x_t, t_tensor, cond) # Pass the tensor 't_tensor'

            # Ensure v_t dtype matches x_t for subtraction
            v_t = v_t.to(x_t.dtype)

            x_t = x_t - dt_tensor * v_t # Use the prepared dt_tensor
            if i % 10 == 0 or i == 1: # Only store some intermediate steps + the last one
                images.append(x_t.clone().cpu()) # Move to CPU to save GPU memory

        # Return only the relevant images (e.g., last one or selected steps)
        return images # List of tensors on CPU

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

def sample_and_save(dataset, rf : Flow, args, result_dir, num_classes):
    # Determine input shape and conditioning...
    if dataset in ["mnist", "cifar"]:
        H = W = 32
        B = 16
        # cond is for embedding lookup, should remain long/int
        cond = torch.arange(0, B).cuda() % num_classes # Use num_classes
        channels = 1 if dataset == "mnist" else 3
    elif dataset == "celeba_hq":
        # H = W = 256 # Original size
        H = W = 64 # Match training size
        B = 16
        channels = 3
        # cond might be needed as input to model, ensure correct dtype later
        # If unconditional but model expects it, create placeholder
        cond = torch.zeros(B, dtype=torch.long).cuda() # Keep long for embedding
    else:
        raise ValueError(f"Dataset {dataset} not supported for sampling")

    # Initialize noise AS BFLOAT16
    x_T = torch.randn(B, channels, H, W,
                      generator=torch.Generator(device='cuda').manual_seed(42),
                      device='cuda',
                      dtype=torch.bfloat16) # <<< FIX: Initialize directly as bfloat16

    # Sample from model - Pass the correct cond dtype
    # sample_fm will handle internal autocast
    images_cpu = rf.sample_fm(x_T, cond, DDIM_NFE=50) # Assuming cond remains long for embedding

    # Save to GIF / PNG (images are already on CPU)
    if images_cpu: # Check if list is not empty
        try:
             # tensors_to_gif expects a list of tensors
             # Ensure make_grid and Image operations are compatible
             from torchvision.utils import make_grid
             import numpy as np

             gif_frames = []
             for img_tensor in images_cpu:
                 # Rescale from [-1, 1] to [0, 1] before saving
                 img_tensor = img_tensor * 0.5 + 0.5
                 img_tensor = img_tensor.clamp(0, 1)
                 # Create grid, convert to numpy, then PIL Image
                 grid = make_grid(img_tensor.float(), nrow=int(B**0.5)) # Adjust nrow
                 img_np = grid.permute(1, 2, 0).numpy() # HWC
                 img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                 gif_frames.append(img_pil)

             if gif_frames:
                 gif_frames[0].save(f"{result_dir}/sample_fm_epoch{args.current_epoch}_step{args.current_step}.gif", # Add epoch/step
                                    save_all=True, append_images=gif_frames[1:],
                                    duration=150, loop=0) # Adjust duration
                 gif_frames[-1].save(f"{result_dir}/sample_fm_epoch{args.current_epoch}_step{args.current_step}_last.png") # Add epoch/step
                 print(f"Saved sample GIF and PNG to {result_dir}")

        except Exception as e:
            print(f"Error saving samples: {e}")
            # print intermediate tensor shapes if needed for debugging
            # print(f"Shape of tensor passed to make_grid: {images_cpu[0].shape}")
    else:
        print("Sampling produced no images.")

    return

# --- In your training loop ---
# Make sure to pass current epoch and step to sample_and_save if you modify it
# e.g.:
# args.current_epoch = epoch + 1
# args.current_step = global_step
# sample_and_save(rf=rf, dataset=args.dataset, args=args, ...)  
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
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

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
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
    parser.add_argument("--save_interval", type=int, default=5, help="interval between model saves")
    args = parser.parse_args()

    result_dir = f"contents/{args.dataset}/epoch_{args.epochs}-batch-size_{args.batch_size}-attn_{args.attn}-iter_{args.iter}-{timestamp}"
    weight_dir = f"weights/{args.dataset}/epoch_{args.epochs}-batch-size_{args.batch_size}-attn_{args.attn}-iter_{args.iter}-{timestamp}"
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
        num_classes = 10
        model = DiT_Llama(
            channels, 32, dim=128, n_layers=10, n_heads=8, num_classes=num_classes, patch_size=patch_size
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
        num_classes = 10
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=num_classes, patch_size=patch_size
        ).cuda()

    elif args.dataset == "celeba_hq":
        data_path = "data/celeba_hq_256"  # path to your flat folder
        transform = transforms.Compose([
            transforms.Resize(64),          # keep original 128x128 or downsample
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        fdatasets = FlatImageDataset(data_path, transform=transform)
        channels = 3
        patch_size = 4  
        num_classes = 1     
        model = DiT_Llama(
        in_channels=3,
        input_size=64,
        patch_size=4,
        dim=768,
        n_layers=8,
        n_heads=8,
        num_classes=1,  # unconditional
    ).cuda()
    model = model.to(dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    rf = Flow(model, iter=args.iter)
    if args.dataset in ["mnist", "cifar"]:
        train_ds = fdatasets(root="./data", train=True, download=True, transform=transform)
    else:
        train_ds = fdatasets
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, drop_last=True,     
        num_workers=4,      # <--- use multiple CPU workers
        pin_memory=True,              # <--- faster CPU→GPU transfers
        persistent_workers=True,      # <--- avoids respawning processes each epoch
        prefetch_factor=2 
    )

    if is_wandb_available:
        wandb.init(project=f"flow", name=f"{args.dataset}_epoch_{args.epochs}-batch-size_{args.batch_size}_attn_{args.attn}-iter_{args.iter}")

    # train model (if there is no model checkpoint)
    optimizer = optim.Adam(rf.model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler()  # optional for bf16
    global_step = 0
    for epoch in range(args.epochs):
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, c in loop:
            x = x.cuda()
            c = c.cuda()
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
                loss = rf.forward_fm(x, c)
            loss.backward()
            optimizer.step()
            # increment step
            global_step += 1
            # update tqdm with step & loss
            loop.set_postfix(loss=loss.item(), step=global_step)
            # log to wandb with global step
            if global_step % args.sample_interval == 0:
                sample_and_save(rf=rf, dataset=args.dataset, args=args, result_dir=result_dir, num_classes=num_classes)
            if epoch % args.save_interval == 0 and global_step == len(train_dl):
                torch.save(rf.model.state_dict(), f"{weight_dir}/model_epoch_{epoch+1}.pth")
            if is_wandb_available:
                wandb.log({"loss": loss.item()}, step=global_step)
    torch.save(rf.model.state_dict(), f"{weight_dir}/final_model.pth")

    sample_and_save(rf=rf, dataset=args.dataset, args=args, result_dir=result_dir, num_classes=num_classes)
