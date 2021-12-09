import tensorflow as tf
import numpy as np


def render_walls(walls, img_size, blur=0.):
    start = walls[:, :, :2]
    end = walls[:, :, 2:4]
    width = walls[:, :, 4]
    probability = walls[:, :, 5]

    center = (start + end) / 2.
    c_x = center[:, :, 0]
    c_y = center[:, :, 1]

    length = calculate_length(start, end)
    angle = calculate_angle(start, end)

    walls_descriptor = tf.stack([c_x, c_y, width, length, angle], axis=-1)
    walls = draw_walls(walls_descriptor, img_size, blur)

    walls = walls * probability[..., tf.newaxis, tf.newaxis]

    return merge(walls)


def calculate_length(start, end):
    delta = start - end
    return tf.sqrt(tf.reduce_sum(tf.square(delta), axis=-1))


def calculate_angle(start, end):
    delta = end - start
    dx = delta[:, :, 0]
    dy = delta[:, :, 1]
    return tf.where(dx*dy > 0.,
                    -tf.math.atan2(dy, dx),
                    tf.math.atan2(tf.abs(dy), tf.abs(dx)))


def draw_walls(walls, img_size, blur):
    x = walls[:, :, 0]
    y = walls[:, :, 1]
    width = walls[:, :, 2]
    length = walls[:, :, 3]
    angle = walls[:, :, 4]

    batch_dim = tf.shape(walls)[0]
    n_walls = tf.shape(walls)[1]

    pixels = get_canvas(img_size, batch_dim, n_walls)
    pixels = rotate_camera(pixels, angle)
    pixels = translate_camera(pixels, x, y, angle)

    xs = pixels[:, :, :, :, 0]
    ys = pixels[:, :, :, :, 1]

    width = width[..., tf.newaxis, tf.newaxis]
    length = length[..., tf.newaxis, tf.newaxis]
    width_condition = blur_function(ys, width, blur)
    length_condition = blur_function(xs, length, blur)
    walls = width_condition * length_condition

    return walls


def get_canvas(img_size, batch_dim, n_walls):
    start = -1.
    delta = 2 / (img_size - 1)
    end = 1. + delta

    pixel_x = tf.range(start, end, delta, dtype=tf.float32)
    pixel_x = tf.repeat(pixel_x[tf.newaxis, ...], img_size, axis=0)
    pixel_x = tf.repeat(pixel_x[tf.newaxis, ...], n_walls, axis=0)
    pixel_x = tf.repeat(pixel_x[tf.newaxis, ...], batch_dim, axis=0)

    pixel_y = tf.range(start, end, delta, dtype=tf.float32)
    pixel_y = tf.repeat(pixel_y[tf.newaxis, ...], img_size, axis=0)
    pixel_y = tf.reverse(pixel_y, axis=[-1])  # [-1..1] -> [1..-1] from top to down
    pixel_y = tf.transpose(pixel_y, perm=[1, 0])
    pixel_y = tf.repeat(pixel_y[tf.newaxis, ...], n_walls, axis=0)
    pixel_y = tf.repeat(pixel_y[tf.newaxis, ...], batch_dim, axis=0)

    pixels = tf.stack([pixel_x, pixel_y], axis=-1)

    return pixels


def rotate_camera(pixels, angle):
    rotation_matrix = tf.stack([tf.stack([tf.math.cos(angle), -tf.math.sin(angle)], axis=-1),
                                tf.stack([tf.math.sin(angle), tf.math.cos(angle)], axis=-1)
                                ], axis=-1)

    rotated_pixels = tf.matmul(pixels, rotation_matrix[:, :, tf.newaxis])

    return rotated_pixels


def translate_camera(pixels, x, y, angle):
    beta = tf.math.atan2(y, x)
    gamma = angle + beta
    c = tf.sqrt(tf.square(x) + tf.square(y))

    x_t = tf.math.cos(gamma) * c
    y_t = tf.math.sin(gamma) * c

    trans_t = tf.stack([x_t, y_t], axis=-1)
    return pixels - trans_t[:, :, tf.newaxis, tf.newaxis, ...]


def blur_function(x, distance, blur):
    if blur == 0.:
        return custom_sign(x+distance/2.) * (1. - custom_sign(x - distance/2.))
    else:
        blur = 1./blur
        v = tf.abs(x) - distance / 2  # v in [-w/2...]
        v = tf.maximum(0., v)  # v in [0...]
        return 0.5 * (1. - tf.cos(tf.exp(-v*blur-tf.math.log(1./np.pi))))


@tf.function
def custom_sign(x):
    """
    x<0 -> 0
    x>=0 -> 1
    """
    return tf.sign(tf.sign(x) + 1.)


def merge(imgs):
    """
    :param imgs: [batch, dim_to_merge, h, w]
    :return:
    """
    return 1. - tf.math.reduce_prod(1. - imgs, axis=1)[..., tf.newaxis]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    w = tf.constant([
        [
            [-0., -0., 1., 0., 0.2, 1.],
            [0., 0., 1., 1., 0.2, 1.],
            [0., 0., 0.1, 1., 0.2, 1.],
            [0., 0., -1, 1, 0.2, 1.],
            [0., 0., -1., 0., 0.2, 1.],
            [0., 0., -1, -1, 0.2, 1.],
            [0., 0., -0.1, -1., 0.2, 1.],
            [0., 0., 1, -1, 0.2, 1.],
        ]
    ], dtype=tf.float32)

    image = render_walls(w, 256, 0.)[0]
    plt.imshow(image, cmap="gray")
    plt.show()
