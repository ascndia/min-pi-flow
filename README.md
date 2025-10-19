# min-Pi-Flow: Minimal Implementation of Pi-Flow

Minimal implementation of [**Pi-Flow**](https://arxiv.org/abs/2510.14974), distillation of flow matching for few-step generation.

This repo provides a minimal implementation to reproduce flow matching distillation results (306LOCs + DiT codebase).

> **⚠️ Warning**: This is an unofficial implementation and is still a work in progress. For the official implementation, please refer to the [Pi-Flow Official Repository](https://github.com/Lakonik/piFlow/tree/main).

**Teacher (Flow Matching, NFE=50)** | **Student (Pi-Flow, NFE=4)**
:---: | :---:
![Teacher FM](https://github.com/enkeejunior1/min-pi-flow/raw/main/contents/mnist/NFE_4-K_8-iter_2/sample_teacher_fm.gif) | ![Student Pi-Flow](https://github.com/enkeejunior1/min-pi-flow/raw/main/contents/mnist/NFE_4-K_8-iter_2/sample_25_pi.gif)
![Teacher FM Last](https://github.com/enkeejunior1/min-pi-flow/raw/main/contents/mnist/NFE_4-K_8-iter_2/sample_teacher_fm_last.png) | ![Student Pi-Flow Last](https://github.com/enkeejunior1/min-pi-flow/raw/main/contents/mnist/NFE_4-K_8-iter_2/sample_25_pi_last.png)

> The left shows the teacher flow matching results, and the right shows the distilled Pi-Flow results (NFE=4).

# Simple Pi-Flow Training

Install torch torchvision einops tqdm (optional wandb)

```bash
conda create -n piflow python=3.11 -y
conda activate piflow
pip install uv 
uv pip install torch torchvision einops tqdm
```

Run

```bash
# MNIST (NFE=4)
python3 train.py --dataset mnist --NFE 4 

# CIFAR-10 (NFE=8)
python3 train.py --dataset cifar --NFE 4
```

> Note: training with NFE=1 tends to be unstable.

---

# Acknowledgments

This implementation is inspired by and heavily based on:
- [CloneofSimo's minRF](https://github.com/cloneofsimo/minRF/tree/main)
- [Pi-Flow Official Implementation](https://github.com/Lakonik/piFlow/tree/main)


---

# Citation

```bibtex
@misc{piflow,
      title={pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation}, 
      author={Hansheng Chen and Kai Zhang and Hao Tan and Leonidas Guibas and Gordon Wetzstein and Sai Bi},
      year={2025},
      eprint={2510.14974},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.14974}, 
}

@inproceedings{gmflow,
  title={Gaussian Mixture Flow Matching Models},
  author={Hansheng Chen and Kai Zhang and Hao Tan and Zexiang Xu and Fujun Luan and Leonidas Guibas and Gordon Wetzstein and Sai Bi},
  booktitle={ICML},
  year={2025},
}
```

If you find this repo helpful and wise enough to cite this repo, please use the following bibtex:

```bibtex
@misc{yong2024minpiflow,
  author       = {Yong-Hyun Park, Mutian Tong, Jiatao Gu},
  title        = {minPiFlow: Minimal Implementation of Pi-flow},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/enkeejunior1/min-pi-flow},
}
```

