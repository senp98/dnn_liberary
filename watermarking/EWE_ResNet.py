


def pairwise_euclid_distance(A):
    sqr_norm_A = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 0)
    sqr_norm_B = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 1)
    inner_prod = tf.matmul(A, A, transpose_b=True)
    tile_1 = tf.tile(sqr_norm_A, [tf.shape(A)[0], 1])
    tile_2 = tf.tile(sqr_norm_B, [1, tf.shape(A)[0]])
    return tile_1 + tile_2 - 2 * inner_prod


def pairwise_cos_distance(A):
    normalized_A = tf.nn.l2_normalize(A, 1)
    return 1 - tf.matmul(normalized_A, normalized_A, transpose_b=True)


def snnl(x, y, t, metric='euclidean'):
    x = tf.nn.relu(x)
    same_label_mask = tf.cast(tf.squeeze(tf.equal(y, tf.expand_dims(y, 1))), tf.float32)
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
    elif metric == 'cosine':
        dist = pairwise_cos_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
    else:
        raise NotImplementedError()
    exp = tf.clip_by_value(tf.exp(-(dist / t)) - tf.eye(tf.shape(x)[0]), 0, 1)
    prob = (exp / (0.00001 + tf.expand_dims(tf.reduce_sum(exp, 1), 1))) * same_label_mask
    loss = - tf.reduce_mean(tf.math.log(0.00001 + tf.reduce_sum(prob, 1)))
    return loss
