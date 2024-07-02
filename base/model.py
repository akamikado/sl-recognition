import tensorflow as tf
import thop
from utils.plotting import feature_visualization
from utils.logging import LOGGER
from utils.misc import time_sync, model_info, fuse_conv_bn, fuse_deconv_and_bn
from modules.conv import Conv, Conv2, DWConv, ConvTranspose


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
        if isinstance(x, dict):
            return self.loss(x)
        return self.predict(x)

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        preds = self.call(batch["img"]) if preds is None else preds
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
            y.append(x if m.f in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(adaptive_avg_pool2d_squeeze(x, (1, 1)))
                if m.i == max(embed):
                    return unbind_concat_embeddings(embeddings)
        return x

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(adaptive_avg_pool2d_squeeze(x, (1, 1)))
                if m.i == max(embed):
                    return unbind_concat_embeddings(embeddings)
        return x

    def _predict_augment(self, x):
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1] and isinstance(
            x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[
            0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(
                f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(
                        m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        bn_layers = [layer for layer in self.layers if isinstance(
            layer, tf.keras.layers.BatchNormalization)]
        return len(bn_layers) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Layer):
                fn(layer)
        return self

    def load(self, weights, verbose=True):
        if isinstance(weights, str):
            self.load_weights(weights)
        else:
            raise ValueError(
                "Unsupported type for 'weights'. Provide a path to a checkpoint file.")

        if verbose:
            LOGGER.info(f"Transferred all weights from {weights}")
