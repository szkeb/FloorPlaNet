import tensorflow as tf
import numpy as np
from diffrend import render_walls


EPSILON = 0.02

RATIO_MIN = 2.5
RATIO_MAX = 8
THIN_MIN_SIZE = 0.04
THIN_MAX_SIZE = 0.2

MIN_DISTANCE = 0.1
MAX_ROTATE = np.math.pi/50.
ZOOM = 1.2


@tf.function
def generate_floorplan(shape, grid_resolution=256):
    batch_dim, n_walls = shape

    thin_width = tf.random.uniform(shape=(batch_dim, 1), minval=THIN_MIN_SIZE, maxval=THIN_MAX_SIZE)
    thin_width = tf.repeat(thin_width, n_walls, axis=1)
    ratio = tf.random.uniform(shape=(batch_dim, 1), minval=RATIO_MIN, maxval=RATIO_MAX)
    thick_width = thin_width * ratio

    wall_idx = tf.range(start=0, limit=n_walls, delta=1)
    wall_idx = tf.repeat(wall_idx[tf.newaxis, ...], batch_dim, axis=0)

    num_of_walls = tf.random.uniform(shape=(batch_dim, 1), minval=0, maxval=n_walls+1, dtype=tf.int32)
    num_of_walls = tf.repeat(num_of_walls, n_walls, axis=1)
    num_of_thick_walls = tf.random.uniform(shape=(batch_dim, 1), minval=0, maxval=n_walls+1, dtype=tf.int32)
    num_of_thick_walls = tf.repeat(num_of_thick_walls, n_walls, axis=1)

    probabilities = tf.where(wall_idx < num_of_walls, 1., 0.)
    widths = tf.where(wall_idx < num_of_thick_walls, thick_width, thin_width)
    angle = tf.random.uniform(shape=(batch_dim, 1), minval=-MAX_ROTATE, maxval=MAX_ROTATE)
    angle = tf.repeat(angle, n_walls, axis=1)

    start_coords, end_coords = get_start_end_points(widths, grid_resolution)
    start_coords, end_coords = rotate(start_coords, end_coords, angle)
    start_coords, end_coords = zoom_in(start_coords, end_coords, zoom=ZOOM)

    start_coords_sorted, end_coords_sorted = sort_coords(start_coords, end_coords)

    start_x = start_coords_sorted[:, :, 0]
    start_y = start_coords_sorted[:, :, 1]
    end_x = end_coords_sorted[:, :, 0]
    end_y = end_coords_sorted[:, :, 1]

    params = tf.stack([start_x, start_y, end_x, end_y, widths], axis=-1)
    params = params * probabilities[..., tf.newaxis]

    walls = tf.concat([params, probabilities[..., tf.newaxis]], axis=-1)
    walls_sorted = sort_walls(walls)

    return walls_sorted


@tf.function
def get_start_end_points(widths, grid_resolution=256):
    batch_dim, n_walls = widths.shape

    free_mx = coords_with_value(get_grid(grid_resolution, batch_dim), 1.)
    connected_mx = coords_with_value(get_grid(grid_resolution, batch_dim), 0.)

    start_coords = tf.zeros(shape=(batch_dim, 1, 2))  # first dummy element
    end_coords = tf.zeros(shape=(batch_dim, 1, 2))  # first dummy element
    for i in range(n_walls):
        first = True if i == 0 else False
        start, from_free = choose_start(free_mx, connected_mx, first)
        end = choose_end(start, from_free, free_mx, connected_mx)
        free_mx, connected_mx = update_mxs(start, end, free_mx, connected_mx, widths[:, i])

        start_coords = tf.concat([start_coords, start[:, tf.newaxis]], axis=1)
        end_coords = tf.concat([end_coords, end[:, tf.newaxis]], axis=1)

    # remove dummy elements
    start_coords = start_coords[:, 1:]
    end_coords = end_coords[:, 1:]

    return start_coords, end_coords


@tf.function
def get_grid(grid_resolution, batch_dim):
    start = -1.
    delta = 2 / (grid_resolution - 1)
    end = 1. + delta

    pixel_x = tf.range(start, end, delta)
    pixel_x = tf.repeat(pixel_x[tf.newaxis, ...], grid_resolution, axis=0)
    pixel_x = tf.repeat(pixel_x[tf.newaxis, ...], batch_dim, axis=0)

    pixel_y = tf.range(start, end, delta)
    pixel_y = tf.repeat(pixel_y[tf.newaxis, ...], grid_resolution, axis=0)
    pixel_y = tf.reverse(pixel_y, axis=[-1])  # [-1..1] -> [1..-1] from top to down
    pixel_y = tf.transpose(pixel_y, perm=[1, 0])
    pixel_y = tf.repeat(pixel_y[tf.newaxis, ...], batch_dim, axis=0)

    pixels = tf.stack([pixel_x, pixel_y], axis=-1)
    return pixels


@tf.function
def choose_start(free_mx, connected_mx, first=False):
    free = np.random.uniform() > 0.5
    if free or first:
        start = random_choose_from(free_mx)
    else:
        start = random_choose_from(connected_mx)

    return start, free


@tf.function
def choose_end(start, from_free, free_mx, connected_mx):
    if from_free:
        current = free_mx
        current = tf.where(hv_line_free(start, free_mx, connected_mx),
                           coords_with_value(current, 1.),
                           coords_with_value(current, 0.))

        end = random_choose_from(current)
    else:
        current = connected_mx
        current = tf.where(hv_line_connected(start, free_mx, connected_mx),
                           coords_with_value(current, 1.),
                           coords_with_value(current, 0.))

        end = random_choose_from(current)

    return end


@tf.function
def random_choose_from(triple_mx):
    batch_dim, H, W, _ = triple_mx.shape
    mask = tf.reshape(triple_mx[:, :, :, -1], shape=(batch_dim, H * W))
    mx_flat = tf.reshape(triple_mx, shape=(batch_dim, H * W, 3))
    coords_flat = mx_flat[:, :, :2]

    random_choose = tf.random.uniform((batch_dim, H*W))
    random_choose = random_choose * mask
    idx = tf.argmax(random_choose, axis=-1)
    choosen_coords = tf.gather(coords_flat, idx, batch_dims=1)

    return choosen_coords


@tf.function
def coords_equivalent(mx, start):
    coords = mx[:, :, :, :2]
    eq = (start[:, tf.newaxis, tf.newaxis, ...] == coords)
    eq = tf.logical_and(eq[:, :, :, -1], eq[:, :, :, -2])
    return eq[:, :, :, tf.newaxis]


@tf.function
def coords_with_value(coord_triples, value):
    batch_dim, H, W, _ = coord_triples.shape
    coords = coord_triples[:, :, :, :2]
    if value == 0.:
        return tf.concat([coords, tf.zeros((batch_dim, H, W, 1))], axis=-1)
    else:
        return tf.concat([coords, tf.ones((batch_dim, H, W, 1))], axis=-1)


@tf.function
def hv_line_free(start, free_mx, connected_mx):
    vertical_condition = tf.logical_and(on_vertical_line(start, free_mx),
                                        not_intersecting(start, free_mx, connected_mx))
    horizontal_condition = tf.logical_and(on_horizontal_line(start, free_mx),
                                          not_intersecting(start, free_mx, connected_mx))
    return tf.logical_or(vertical_condition, horizontal_condition)[:, :, :, tf.newaxis]


@tf.function
def hv_line_connected(start, free_mx, connected_mx):
    vertical_condition = tf.logical_and(on_vertical_line(start, connected_mx),
                                        not_inside(start, free_mx, connected_mx))
    horizontal_condition = tf.logical_and(on_horizontal_line(start, connected_mx),
                                          not_inside(start, free_mx, connected_mx))
    return tf.logical_or(vertical_condition, horizontal_condition)[:, :, :, tf.newaxis]


@tf.function
def on_vertical_line(start, mx, min_distance=MIN_DISTANCE):
    x = mx[:, :, :, 0]
    y = mx[:, :, :, 1]
    start_x = start[:, 0][:, tf.newaxis, tf.newaxis, ...]
    start_y = start[:, 1][:, tf.newaxis, tf.newaxis, ...]
    on_line = x == start_x
    distance = tf.abs(start_y - y)
    return tf.logical_and(on_line, distance >= min_distance)


@tf.function
def on_horizontal_line(start, mx, min_distance=MIN_DISTANCE):
    x = mx[:, :, :, 0]
    y = mx[:, :, :, 1]
    start_x = start[:, 0][:, tf.newaxis, tf.newaxis, ...]
    start_y = start[:, 1][:, tf.newaxis, tf.newaxis, ...]
    on_line = y == start_y
    distance = tf.abs(start_x-x)
    return tf.logical_and(on_line, distance >= min_distance)


@tf.function
def not_intersecting(start, free_mx, connected_mx):
    return tf.where(free_mx[:, :, :, -1] != 0., True, False)


@tf.function
def not_inside(start, free_mx, connected_mx):
    return True


@tf.function
def update_mxs(start, end, free_mx, connected_mx, width):
    free_coords = free_mx[:, :, :, :2]
    free_x = free_coords[..., 0]
    free_y = free_coords[..., 1]

    start = start[:, tf.newaxis, tf.newaxis, ...]
    end = end[:, tf.newaxis, tf.newaxis, ...]

    top = tf.maximum(start[..., 1], end[..., 1])
    bottom = tf.minimum(start[..., 1], end[..., 1])
    left = tf.minimum(start[..., 0], end[..., 0])
    right = tf.maximum(start[..., 0], end[..., 0])

    w2 = width[:, tf.newaxis, tf.newaxis, ...]/2.

    condition = tf.where((start[:, :, :, 0] == end[:, :, :, 0])[..., tf.newaxis],
                         tf.logical_and(between(bottom, free_y, top)[:, :, :, tf.newaxis],
                                        between(left - w2, free_x, right + w2)[:, :, :, tf.newaxis]),
                         tf.logical_and(between(bottom - w2, free_y, top + w2)[:, :, :, tf.newaxis],
                                        between(left, free_x, right)[:, :, :, tf.newaxis]))

    free_updated = tf.where(condition,
                            coords_with_value(free_mx, 0.),
                            free_mx)

    connected_updated = tf.where(condition,
                                 coords_with_value(connected_mx, 1.),
                                 connected_mx)

    return free_updated, connected_updated


@tf.function
def between(lower, value, upper):
    return tf.logical_and(lower <= value, value <= upper)


@tf.function
def zoom_in(start_coords, end_coords, zoom):
    return start_coords*zoom, end_coords*zoom


@tf.function
def rotate(start_coords, end_coords, angle):
    rotation_matrix = tf.stack([tf.stack([tf.math.cos(angle), -tf.math.sin(angle)], axis=-1),
                                tf.stack([tf.math.sin(angle), tf.math.cos(angle)], axis=-1)
                                ], axis=-1)

    start_rotated = tf.matmul(start_coords[:, :, tf.newaxis, ...], rotation_matrix)
    end_rotated = tf.matmul(end_coords[:, :, tf.newaxis, ...], rotation_matrix)

    return start_rotated[:, :, 0, :], end_rotated[:, :, 0, :]


@tf.function
def distance(coordinate):
    x = coordinate[..., 0]
    y = coordinate[..., 1]
    return tf.sqrt(x*x + y*y)


@tf.function
def sort_coords(start_coords, end_coords):
    # shape = [B, NW, 4]

    # transform from [-1, 1] to [0, 2]
    sc = start_coords + 1.
    ec = end_coords + 1.

    # distance from the upper left corner
    sd = distance(sc)[..., tf.newaxis]
    ed = distance(ec)[..., tf.newaxis]

    # swap if necessary
    start_final = tf.where(sd <= ed, start_coords, end_coords)
    end_final = tf.where(sd > ed, start_coords, end_coords)

    return start_final, end_final


@tf.function
def sort_walls(walls):
    # shape = [B, NW, 6]

    starts = walls[:, :, :2] + 1.
    probs = walls[:, :, -1]

    distances = distance(starts)
    distances = tf.where(probs == 0., 10., distances)

    order = tf.argsort(distances, axis=-1)
    walls = tf.gather(walls, order, batch_dims=1)
    return walls