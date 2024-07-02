import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from logging import LOGGER
from pathlib import Path


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:
            return
    if isinstance(x, tf.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / \
                f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            # select batch index 0, block by channels
            blocks = tf.split(x[0], channels, axis=0)
            n = min(n, channels)  # number of plots
            _, ax = plt.subplots(math.ceil(n / 8), 8,
                                 tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save
