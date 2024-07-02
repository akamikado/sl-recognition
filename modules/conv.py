import tensorflow as tf
import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(tf.keras.layers.Layer):
    default_act = tf.keras.activations.silu()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            c2, kernel_size=k, strides=s, padding='same', dilation_rate=d, groups=g, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization()

        if act is True:
            self.act = tf.keras.layers.Activation('relu')
        elif isinstance(act, tf.keras.layers.Layer):
            self.act = act
        else:
            self.act = tf.keras.layers.Activation('linear')

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv2(Conv):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.conv = tf.keras.layers.Conv2D(c2, kernel_size=1, strides=s,
                                           padding='same', dilation_rate=d, groups=g, use_bias=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        weight_shape = self.conv.weights[0].shape
        w = tf.zeros(shape=weight_shape, dtype=self.conv.weights[0].dtype)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class ConvTranspose(tf.keras.layers.Layer):
    """Convolution transpose 2d layer."""

    default_act = tf.keras.activations.silu()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding='same' if p == 0 else 'valid',
            use_bias=not bn  # Disable bias if BatchNormalization is enabled
        )

        # Define BatchNormalization or Identity layer based on 'bn' flag
        self.bn = tf.keras.layers.BatchNormalization() if bn else tf.keras.layers.Identity()

        # Define activation function
        self.act = act if isinstance(
            act, tf.keras.layers.Layer) else tf.keras.layers.Activation('linear')

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))

# TODO
# class RepVGGDW(tf.keras.layers.Layer):
# class RepConv(tf.keras.layers.Layer):
