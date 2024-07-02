import tensorflow as tf
from utils.plotting import feature_visualization


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
            y.apped(x if m.f in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(adaptive_avg_pool2d_squeeze(x, (1, 1)))
                if m.i == max(embed):
                    return unbind_concat_embeddings(embeddings)
        return x
