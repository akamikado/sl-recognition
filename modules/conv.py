import tensorflow as tf
import math
import numpy as np
from utils.misc import fuse_conv_bn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(tf.keras.layers.Layer):
    default_act = tf.keras.activations.silu()

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

    default_act = tf.keras.activations.silu()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding='same' if p == 0 else 'valid',
            use_bias=not bn
        )

        self.bn = tf.keras.layers.BatchNormalization() if bn else tf.keras.layers.Identity()

        self.act = act if isinstance(
            act, tf.keras.layers.Layer) else tf.keras.layers.Activation('linear')

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        return self.act(self.conv_transpose(x))


class RepVGGDW(tf.keras.layers.Layer):

    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = tf.keras.activations.silu()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @tf.function
    def fuse(self):
        conv = fuse_conv_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = tf.pad(conv1_w, [[2, 2], [2, 2], [0, 0], [0, 0]])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class RepConv(tf.keras.layers.Layer):
    default_act = tf.keras.activations.silu()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(
            act, tf.keras.layers.Layer) else tf.keras.layers.Identity()

        self.bn = tf.keras.layers.BatchNormalization(
        ) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(kernel1x1, paddings=[[1, 1], [1, 1], [0, 0], [0, 0]])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, tf.keras.layers.BatchNormalization):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros(
                    (self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = tf.convert_to_tensor(kernel_value)
                with tf.device(branch.weights[0].device):
                    self.id_tensor = tf.identity(self.id_tensor)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = tf.keras.layers.Conv2D(
            filters=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            strides=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation_rate=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            use_bias=True
        )
        self.conv.trainable = False
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
