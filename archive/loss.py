import tensorflow as tf
import numpy


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    if xywh:
        x1, y1, w1, h1 = tf.split(box1, 4, axis=-1)
        x2, y2, w2, h2 = tf.split(box2, 4, axis=-1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(box1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(box2, 4, axis=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = tf.math.maximum(b1_x2, b2_x2) - tf.math.maximum(b1_x1, b2_x1)
    inter = tf.clip_by_value(inter, 0, tf.reduce_max(inter))
    inter *= tf.math.maximum(b1_y2, b2_y2) - tf.math.maximum(b1_y1, b2_y1)
    inter = tf.clip_by_value(inter, 0, tf.reduce_max(inter))

    union = (w1 * h1) + (w2 * h2) - inter + eps

    iou = inter / union
    cw = tf.math.maximum(b1_x2, b2_x2) - tf.math.minimum(b1_x1, b2_x1)
    ch = tf.math.maximum(b1_y2, b2_y2) - tf.math.minimum(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / (tf.math.pi ** 2)) * \
        (tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1)) ** 2
    alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = tf.split(bbox, 2, axis=-1)
    return tf.concat([anchor_points - x1y1, x2y2 - anchor_points], axis=-1)


def df_loss(pred_dist, target):
    tl = tf.cast(target, tf.int32)
    tr = tl + 1
    wl = tf.cast(tr, numpy.float32) - target
    wr = 1 - wl

    loss_left = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tl, logits=pred_dist)
    loss_right = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tr, logits=pred_dist)

    return loss_left * wl + loss_right * wr


def compute_bbox_loss(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, reg_max, use_dfl=False):
    weight = tf.reduce_sum(target_scores, axis=-1)[fg_mask]
    weight = tf.expand_dims(weight, axis=-1)

    iou = bbox_iou(pred_bboxes[fg_mask],
                   target_bboxes[fg_mask], xywh=False)
    loss_iou = tf.reduce_sum((1.0 - iou) * weight) / target_scores_sum

    if use_dfl:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max)
        loss_dfl = df_loss(pred_dist[fg_mask], target_ltrb[fg_mask]) * weight
        loss_dfl = tf.reduce_sum(loss_dfl) / target_scores_sum
    else:
        loss_dfl = tf.constant(0.0, dtype=pred_dist.dtype)

    return loss_iou, loss_dfl


def compute_cls_loss(preds, true_labels):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        true_labels, preds, from_logits=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_labels, logits=preds)

    loss_mean = tf.reduce_mean(loss)
    return loss_mean, loss


def DFL(c1=16):
    conv = tf.keras.layers.Conv2D(1, (1, 1), use_bias=False, trainable=False)
    x = tf.range(c1, dtype=tf.float32)
    conv.build((None, c1, None, None))
    conv.kernel.assign(tf.reshape(x, (1, c1, 1, 1)))

    def apply_dfl(x):
        b, _, a = x.shape  # batch, channels, anchors
        x = tf.transpose(tf.reshape(x, [b, 4, c1, a]), [0, 2, 1, 3])
        x = tf.nn.softmax(x, axis=1)
        x = conv(x)
        return tf.reshape(x, [b, 4, a])

    return apply_dfl
