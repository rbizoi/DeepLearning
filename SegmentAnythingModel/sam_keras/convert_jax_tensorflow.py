import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops

_NMS_TILE_SIZE = 512

def _self_suppression(in_args):
    iou, _, iou_sum = in_args
    batch_size = tf.shape(iou)[0]
    can_suppress_others = tf.reshape(
        tf.reduce_max(iou, axis=1) <= 0.5, [batch_size, -1, 1]
    )
    can_suppress_others = tf.cast(can_suppress_others, iou.dtype)
    iou_suppressed = (
        tf.reshape(
            (tf.reduce_max(can_suppress_others * iou, 1) <= 0.5), [batch_size, -1, 1],
        )
        * iou
    )
    iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
    return iou_suppressed, tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new


def _cross_suppression(in_args):
    boxes, box_slice, iou_threshold, inner_idx = in_args
    batch_size = tf.shape(boxes)[0]
    new_slice = array_ops.slice(
        boxes, [0, inner_idx * _NMS_TILE_SIZE, 0], [batch_size, _NMS_TILE_SIZE, 4]
    )
    iou = _bbox_overlap(new_slice, box_slice)
    ret_slice = (
        tf.expand_dims((tf.reduce_all(iou < iou_threshold, [1])).astype(box_slice.dtype), 2)
        * box_slice
    )
    return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(in_args):
    boxes, iou_threshold, output_size, idx = in_args
    num_tiles = tf.shape(boxes)[1] // _NMS_TILE_SIZE
    batch_size = tf.shape(boxes)[0]

    box_slice = array_ops.slice(
        boxes, [0, idx * _NMS_TILE_SIZE, 0], [batch_size, _NMS_TILE_SIZE, 4]
    )

    def _loop_cond(in_args):
        _, _, _, inner_idx = in_args
        return inner_idx < idx

    _, box_slice, _, _ = tf.while_loop(
        _loop_cond, _cross_suppression, (boxes, box_slice, iou_threshold, 0)
    )

    iou = _bbox_overlap(box_slice, box_slice)
    mask = tf.expand_dims(
        tf.reshape(
            math_ops.range(_NMS_TILE_SIZE), [1, -1]
        )
        > tf.reshape(math_ops.range(_NMS_TILE_SIZE), [-1, 1]),
        0,
    )
    iou *= (tf.logical_and(mask, iou >= iou_threshold)).astype(iou.dtype)

    def _loop_cond2(in_args):
        _, loop_condition, _ = in_args
        return loop_condition

    suppressed_iou, _, _ = tf.while_loop(
        _loop_cond2, _self_suppression, (iou, True, tf.reduce_sum(iou, [1, 2]))
    )
    suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
    box_slice *= tf.expand_dims(1.0 - suppressed_box.astype(box_slice.dtype), 2)

    mask = tf.reshape(
        (tf.equal(math_ops.range(num_tiles), idx)).astype(boxes.dtype), [1, -1, 1, 1]
    )
    boxes = tf.tile(
        tf.expand_dims(box_slice, 1), [1, num_tiles, 1, 1]
    ) * mask + tf.reshape(boxes, [batch_size, num_tiles, _NMS_TILE_SIZE, 4]) * (
        1 - mask
    )
    boxes = tf.reshape(boxes, [batch_size, -1, 4])

    output_size += tf.reduce_sum(tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
    return boxes, iou_threshold, output_size, idx + 1


def non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold):
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    pad = tf.cast(
        tf.math.ceil(tf.cast(num_boxes, tf.float32) / _NMS_TILE_SIZE), tf.int32
    ) * _NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(boxes, [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(scores, [[0, 0], [0, pad]])
    num_boxes += pad

    def _loop_cond(in_args):
        unused_boxes, unused_threshold, output_size, idx = in_args
        return tf.math.logical_and(
            tf.reduce_min(output_size) < max_output_size,
            idx < num_boxes // _NMS_TILE_SIZE,
        )

    selected_boxes, _, output_size, _ = tf.while_loop(
        _loop_cond,
        _suppression_loop_body,
        (boxes, iou_threshold, tf.zeros([batch_size], tf.int32), 0),
    )
    idx = num_boxes - tf.math.top_k(
        tf.cast(
            tf.reduce_all(selected_boxes > 0,
