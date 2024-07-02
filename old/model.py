import tensorflow as tf
from tensorflow.keras import layers, models
from loss import DFL


def conv_layer(input, filters, k, s, p):
    if p == 0:
        x = layers.Conv2D(filters=filters, kernel_size=(k, k),
                          strides=k, padding='valid')(input)
    elif p == 1:
        x = layers.Conv2D(filters=filters, kernel_size=(k, k),
                          strides=s, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('silu')(x)

    return x


def bottleneck_block(input, shortcut):
    c = input.shape[-1]
    x = conv_layer(input=input, filters=c, k=3,
                   s=1, p=1)
    x = conv_layer(input=x, filters=c, k=3,
                   s=1, p=1)

    if shortcut:
        x = layers.Add()([x, input])

    return x


def c2f_block(input, filters, n, shortcut=False):
    x = conv_layer(input=input, filters=filters, k=1, s=1, p=0)

    x1, x2 = tf.split(x, 2, axis=-1)

    x = layers.Concatenate(axis=-1)([x1, x2])

    b = bottleneck_block(input=x2, shortcut=shortcut)
    x = layers.Concatenate(axis=-1)([x, b])
    for _ in range(n-1):
        b = bottleneck_block(input=b, shortcut=shortcut)
        x = layers.Concatenate(axis=-1)([x, b])

    x = conv_layer(input=x, filters=filters, k=1,
                   s=1, p=0)

    return x


def sppf_block(input, filters):
    x = conv_layer(input=input, filters=filters, k=1, s=1, p=0)

    x1 = layers.MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)
    x2 = layers.MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)
    x3 = layers.MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(x)

    x = layers.Concatenate(axis=-1)([x, x1, x2, x3])

    x = conv_layer(input=x, filters=filters, k=1, s=1, p=0)

    return x


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points = []
    stride_tensor = []
    for i, stride in enumerate(strides):
        _, h, w, _ = feats[i].shape
        sx = tf.range(w, dtype=tf.float32) + grid_cell_offset  # shift x
        sy = tf.range(h, dtype=tf.float32) + grid_cell_offset  # shift y
        sy, sx = tf.meshgrid(sy, sx)
        anchor_points.append(tf.stack((sx, sy), axis=-1))
        stride_tensor.append(tf.ones((h, w, 1), dtype=tf.float32) * stride)
    anchor_points = tf.concat([tf.reshape(a, (-1, 2))
                              for a in anchor_points], axis=0)
    stride_tensor = tf.concat([tf.reshape(s, (-1, 1))
                              for s in stride_tensor], axis=0)
    return anchor_points, stride_tensor


def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
    lt, rb = tf.split(distance, 2, axis=axis)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return tf.concat([c_xy, wh], axis)  # xywh bbox
    return tf.concat([x1y1, x2y2], axis)  # xyxy bbox


def decode_bboxes(bboxes, anchors, end2end=False):
    return dist2bbox(bboxes, anchors, xywh=not end2end, axis=1)


def detect_block(inputs, nc, ch, reg_max, training=False):

    no = nc + reg_max * 4
    stride = tf.Variable(tf.zeros(len(ch)), trainable=False)

    cv2_layers = [tf.keras.Sequential([
        conv_layer(filters=max(16, ch[0] // 4, reg_max*4), k=3, s=1, p=1),
        conv_layer(filters=max(16, ch[0] // 4, reg_max*4), k=3, s=1, p=1),
        layers.Conv2D(filters=4 * reg_max, kernel_size=(1, 1),
                      strides=1, padding='same')
    ]) for x in ch]
    cv3_layers = [tf.keras.Sequential([
        conv_layer(filters=max(ch[0], min(nc, 100)), k=3, s=1, p=1),
        conv_layer(filters=max(ch[0], min(nc, 100)), k=3, s=1, p=1),
        layers.Conv2D(filters=nc, kernel_size=(1, 1),
                      strides=1, padding='same')
    ]) for x in ch]

    dfl_fn = DFL(reg_max) if reg_max > 1 else tf.identity

    x = inputs
    for i in range(len(ch)):
        x[i] = tf.concat([cv2_layers[i](x[i]), cv3_layers[i](x[i])], axis=-1)

    if training:
        return x

    shape = tf.shape(x[0])
    x_cat = tf.concat([tf.reshape(xi, [shape[0], no, -1]) for xi in x], axis=2)

    anchors, strides = make_anchors(x, stride, 0.5)

    box, cls = tf.split(x_cat, [reg_max * 4, nc], axis=1)
    dbox = decode_bboxes(dfl_fn(box), anchors) * strides

    y = tf.concat([dbox, tf.nn.sigmoid(cls)], axis=1)

    return y

# model d(depth_multiple)  w(width_multiple)  r(ratio)
# n     0.33                0.25                2.0
# s     0.33                0.50                2.0
# l     0.67                0.75                1.5
# m     1.00                1.00                1.0
# x     1.00                1.25                1.0


def YoloV8(w, d, r, nc, ch, reg_max):
    input = layers.Input(shape=(640, 640, 3))

    # Backbone of YoLoV8

    x = conv_layer(input=input, filters=64*w, k=3, s=2, p=1)

    x = conv_layer(input=input, filters=128*w, k=3, s=4, p=1)

    x = c2f_block(input=x, filters=128*w, n=round(3*d), shortcut=True)

    x = conv_layer(input=input, filters=256*w, k=3, s=8, p=1)

    x = c2f_block(input=x, filters=256*w, n=round(6*d), shortcut=True)

    x1 = x

    x = conv_layer(input=input, filters=512*w, k=3, s=16, p=1)

    x = c2f_block(input=x, filters=512*w, n=round(6*d), shortcut=True)

    x2 = x

    x = conv_layer(input=input, filters=512*w*r, k=3, s=32, p=1)

    x = c2f_block(input=x, filters=512*w*r, n=round(3*d), shortcut=True)

    x = sppf_block(input=x, filters=512*w*r)

    # Head of YoLoV8

    x3 = layers.UpSampling2D(size=(2, 2))(x)

    x2 = layers.Concatenate(axis=-1)([x2, x3])

    x2 = c2f_block(input=x2, filters=512*w, n=round(3*d), shortcut=False)

    x4 = x2

    x2 = layers.UpSampling2D(size=(2, 2))(x2)

    x1 = layers.Concatenate(axis=-1)([x1, x2])

    x1 = c2f_block(input=x1, filters=256*w, n=round(3*d), shortcut=False)

    output1 = x1

    x1 = conv_layer(input=x1, filters=256*w, k=3, s=2, p=1)

    x1 = layers.Concatenate(axis=-1)([x1, x4])

    x1 = c2f_block(input=x1, filters=512*w, n=round(3*d), shortcut=False)

    output2 = x1

    x1 = conv_layer(input=x1, filters=512*w, k=3, s=2, p=1)

    x = layers.Concatenate(axis=-1)([x1, x])

    x = c2f_block(input=x, filters=512*w*r, n=round(3*d), shortcut=False)

    output3 = x

    outputs = detect_block(
        inputs=[output1, output2, output3], nc=nc, ch=ch, reg_max=reg_max, training=False)

    model = models.Model(inputs=input, outputs=outputs)

    return model
