import tensorflow as tf
import numpy as np
import math
from pathlib import Path
import matplotlig.pyplot as plt


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


def adaptive_avg_pool2d_squeeze(input_tensor):
    x = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    x = tf.squeeze(x, axis=-1)
    return x


def unbind_concat_embeddings(embeddings):
    concatenated = tf.concat(embeddings, axis=1)
    num_splits = concatenated.shape[0]
    unbound = tf.split(concatenated, num_splits, axis=0)

    return unbound


class BaseModel(tf.keras.layers.Layer):
    def call(self, x):
        if isInstance(x, dict):
            return self.loss(x)
        return self.predict(x)

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        prods = self.call(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def predict(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]
            if profile:
                self.__profile_one_layer(m, x, dt)
            x = m(x)
            y.apped(x if m.f in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(adaptive_avg_pool2d_squeeze(x, (1, 1)))
                if m.i == max(embed):
                    return unbind_concat_embeddings(embeddings)
        return x
