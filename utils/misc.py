import tensorflow as tf
import time
import thop
from copy import deepcopy
from pathlib import Path
from logging import LOGGER


def time_sync():
    """TensorFlow-accurate time."""
    if tf.config.list_physical_devices('GPU'):
        tf.experimental.sync_devices()  # Synchronize all devices
    return time.time()


def model_info(model, detailed=False, verbose=True, imgsz=640):
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype)
            )

    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(
        model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(
        f"{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops(model, imgsz=640):
    if not thop:
        return 0.0

    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:
            stride = max(int(model.stride.max()), 32) if hasattr(
                model, "stride") else 32  # max stride
            im = tf.zeros((1, p.shape[1], stride, stride), dtype=p.dtype)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[
                0] / 1e9 * 2
            return flops * imgsz[0] / stride * imgsz[1] / stride
        except Exception:
            im = tf.zeros((1, p.shape[1], *imgsz), dtype=p.dtype)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
    except Exception:
        return 0.0


def de_parallel(model):
    return model.module if is_parallel(model) else model


def is_parallel(model):
    return isinstance(model, tf.distribute.Strategy)


def fuse_conv_bn(conv, bn):
    fusedconv = tf.keras.layers.Conv2D(
        filters=conv.out_channels,
        kernel_size=conv.kernel_size,
        strides=conv.stride,
        padding=conv.padding,
        dilation_rate=conv.dilation,
        groups=conv.groups,
        use_bias=True
    )

    fusedconv.trainable = False

    fusedconv.build(conv.weight.shape)

    w_conv = tf.reshape(conv.weight, (conv.out_channels, -1))
    w_bn = tf.linalg.diag(bn.weight / tf.sqrt(bn.epsilon + bn.moving_variance))
    fusedconv.set_weights([tf.matmul(w_bn, w_conv)])

    b_conv = tf.zeros(conv.weight.shape[0], dtype=conv.weight.dtype)
    if conv.bias is not None:
        b_conv = conv.bias
    b_bn = bn.bias - bn.weight * \
        (bn.moving_mean / tf.sqrt(bn.moving_variance + bn.epsilon))
    fused_bias = tf.squeeze(
        tf.matmul(w_bn, tf.expand_dims(b_conv, axis=-1))) + b_bn
    fusedconv.bias = fused_bias.numpy()

    return fusedconv


def fuse_deconv_and_bn(deconv, bn):
    fused_deconv = tf.keras.layers.Conv2DTranspose(
        filters=deconv.out_channels,
        kernel_size=deconv.kernel_size,
        strides=deconv.stride,
        padding=deconv.padding,
        output_padding=deconv.output_padding,
        dilation_rate=deconv.dilation,
        groups=deconv.groups,
        use_bias=True
    )

    fused_deconv.trainable = False

    fused_deconv.build(deconv.weight.shape)

    w_deconv = tf.reshape(deconv.weight, (deconv.out_channels, -1))
    w_bn = tf.linalg.diag(bn.weight / tf.sqrt(bn.epsilon + bn.moving_variance))
    fused_deconv.set_weights([tf.matmul(w_bn, w_deconv)])

    b_deconv = tf.zeros(deconv.output_channels, dtype=deconv.weight.dtype)
    if deconv.bias is not None:
        b_deconv = deconv.bias
    b_bn = bn.bias - bn.weight * \
        (bn.moving_mean / tf.sqrt(bn.moving_variance + bn.epsilon))
    fused_bias = tf.squeeze(
        tf.matmul(w_bn, tf.expand_dims(b_deconv, axis=-1))) + b_bn
    fused_deconv.bias = fused_bias.numpy()

    return fused_deconv
