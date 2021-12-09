from preprocess import separate_elements, read_and_invert
from grid import create_grid
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from generator import distance
from diffrend import render_walls
from net import FloorPlaNet
from train import IMG_SIZE, PARAM_SHAPE
from doors import find_doors


def postprocess_params(params, doors, nrows, ncols, window_size, min_area, delta_absolute, merge_delta, door_delta):
    walls = get_absolute_position(params, nrows, ncols, window_size)
    walls = tf.reshape(walls, shape=(nrows * ncols * 4, -1))
    walls = filter_not_visible(walls)
    original = walls

    walls = filter_small_walls(walls, min_area)
    without_small = walls

    walls = force_grid_windows(walls)
    force_window = walls

    walls = correct_sides(walls)
    walls = force_grid_absolute(walls, delta=delta_absolute)
    force_abs = walls

    walls = filter_points(walls)
    walls = filter_same(walls)
    filtered = walls

    walls = merge_walls(walls, merge_delta)
    merged = walls
    walls = fix_corners(walls)
    corners = walls
    final, doors = snap_doors(walls, doors, door_delta)
    return [original, without_small, force_window, force_abs, filtered, merged, corners, final], doors


def snap_doors(walls, doors, delta):
    walls = walls.numpy()
    nearby_walls = [near(d, walls, delta) for d in doors]
    doors = np.asarray([d for i, d in enumerate(doors) if len(nearby_walls[i]) > 1])
    nearby_walls = np.asarray(nw for nw in nearby_walls if len(nw) > 1)

    if len(doors) > 0:
        for i, (d, nw) in enumerate(zip(doors, nearby_walls)):
            w1 = nw[0]
            w2 = nw[1]

            widx1 = w1[0]
            widx2 = w2[0]

            offset1 = w1[1]*2
            offset2 = w2[1]*2

            avgx = (walls[widx1][offset1] + walls[widx2][offset2]) / 2.
            avgy = (walls[widx1][offset1+1] + walls[widx2][offset2+1]) / 2.

            doors[i] = [avgx, avgy]

    return walls, doors


def near(door, walls, delta):
    nearby = []
    for i, w in enumerate(walls):
        if distance_np(door, w[:2]) < delta:
            nearby.append([i, 0])
        if distance_np(door, w[2:4]) < delta:
            nearby.append([i, 1])

    return np.asarray(nearby)


def filter_points(walls):
    walls = walls.numpy()
    filtered = [w for w in walls if distance_np(w[:2], w[2:4]) > 0.001]
    return tf.constant(filtered, dtype=tf.float32)


def filter_same(walls):
    walls = walls.numpy()

    filtered = []
    for i, w1 in enumerate(walls):
        ok = True
        for j, w2 in enumerate(walls[i+1:]):
            coords1 = np.sort(w1[:4])
            coords2 = np.sort(w2[:4])
            if np.sum(coords1-coords2, axis=-1) == 0.:
                ok = False
        if ok:
            filtered.append(w1)

    return tf.constant(filtered, dtype=tf.float32)


def correct_sides(walls):
    start = walls[..., :2]
    end = walls[..., 2:4]
    width = walls[..., 4][..., tf.newaxis]
    probs = walls[..., -1]
    dir = start - end
    center = (start - end) / 2.
    dirdis = distance(dir)[..., tf.newaxis]
    per = tf.stack([dir[..., 1], -dir[..., 0]], axis=-1)
    perdis = distance(per)[..., tf.newaxis]
    per = tf.where(perdis > 0., per/perdis, per)

    start = tf.where(width > dirdis,
                     center + per * width/2.,
                     start)
    end = tf.where(width > dirdis,
                   center - per * width/2.,
                   end)
    width = tf.where(width > dirdis,
                     dirdis,
                     width)

    return tf.concat([start, end, width, probs[..., tf.newaxis]], axis=-1)


def fix_corners(walls, delta=0.001):
    walls = walls.numpy()
    walls_fixed = list(walls)
    for i in range(len(walls)):
        for j in range(len(walls)):
            if i != j:
                w1 = walls[i]
                w2 = walls[j]
                start1 = w1[:2]
                start2 = w2[:2]
                end1 = w1[2:4]
                end2 = w2[2:4]
                width1 = w1[4]
                width2 = w2[4]

                if distance_np(start1, start2) < delta and not on_the_same_line(end1, end2):
                    dir1 = start1 - end1
                    dir2 = start2 - end2
                    dir1 = dir1 / distance(dir1)
                    dir2 = dir2 / distance(dir2)

                    start1_fixed = start1 + dir1 * width2 / 2.
                    start2_fixed = start2 + dir2 * width1 / 2.

                    walls_fixed[i][:2] = start1_fixed
                    walls_fixed[j][:2] = start2_fixed

                if distance_np(start1, end2) < delta and not on_the_same_line(end1, start2):
                    dir1 = start1 - end1
                    dir2 = end2 - start2
                    dir1 = dir1 / distance(dir1)
                    dir2 = dir2 / distance(dir2)

                    start1_fixed = start1 + dir1 * width2 / 2.
                    end2_fixed = end2 + dir2 * width1 / 2.

                    walls_fixed[i][:2] = start1_fixed
                    walls_fixed[j][2:4] = end2_fixed

                if distance_np(end1, start2) < delta and not on_the_same_line(start1, end2):
                    dir1 = end1 - start1
                    dir2 = start2 - end2
                    dir1 = dir1 / distance(dir1)
                    dir2 = dir2 / distance(dir2)

                    end1_fixed = end1 + dir1 * width2 / 2.
                    start2_fixed = start2 + dir2 * width1 / 2.

                    walls_fixed[i][2:4] = end1_fixed
                    walls_fixed[j][:2] = start2_fixed

                if distance_np(end1, end2) < delta and not on_the_same_line(start1, start2):
                    dir1 = end1 - start1
                    dir2 = end2 - start2
                    dir1 = dir1 / distance(dir1)
                    dir2 = dir2 / distance(dir2)

                    end1_fixed = end1 + dir1 * width2 / 2.
                    end2_fixed = end2 + dir2 * width1 / 2.

                    walls_fixed[i][2:4] = end1_fixed
                    walls_fixed[j][2:4] = end2_fixed

    return tf.constant(walls_fixed, dtype=tf.float32)


def merge_walls(walls, delta):
    walls = walls.numpy()

    joining = find_joining(walls, delta)
    there_is_mergeable = joining[0]
    while there_is_mergeable:
        i, j, merged = joining[1:]
        walls = np.delete(walls, [i, j], axis=0)
        walls = np.append(walls, np.expand_dims(merged, axis=0), axis=0)
        joining = find_joining(walls, delta)
        there_is_mergeable = joining[0]

    return tf.constant(walls, dtype=tf.float32)


def find_joining(walls, delta):
    for i, wall in enumerate(walls):
        start = wall[:2]
        end = wall[2:4]
        for j, other in enumerate(walls):
            if i != j:
                start_other = other[:2]
                end_other = other[2:4]
                width = (wall[4] + other[4]) / 2.
                dis1 = distance_np(start, start_other)
                dis2 = distance_np(start, end_other)
                dis3 = distance_np(end, start_other)
                dis4 = distance_np(end, end_other)

                if ((dis1 < delta and on_the_same_line(end, end_other)) or
                    (dis2 < delta and on_the_same_line(end, start_other)) or
                    (dis3 < delta and on_the_same_line(start, end_other)) or
                    (dis4 < delta and on_the_same_line(start, start_other))):
                    merged_wall = [*furthest([start, end, start_other, end_other]), width, 1.]
                    return True, i, j, merged_wall

    return False, _


def furthest(points):
    distances = []
    for p1 in points:
        for p2 in points:
            distances.append(distance_np(p1, p2))

    biggest = np.argmax(distances, axis=-1)
    idx1 = biggest // 4
    idx2 = biggest % 4

    return [*points[idx1], *points[idx2]]


def distance_np(point_1, point_2):
    dif = point_1 - point_2
    length = np.sqrt(dif[0]**2 + dif[1]**2)
    return length


def on_the_same_line(point1, point2, delta=0.005):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    return abs(x1 - x2) < delta or abs(y1 - y2) < delta


def filter_small_walls(walls, min_area):
    start = walls[..., :2]
    end = walls[..., 2:4]
    width = walls[..., 4]
    length = distance(start - end)
    area = width * length

    walls = tf.constant([w for w, a in zip(walls.numpy(), area.numpy()) if a >= min_area])
    return walls


def translate_origo(walls):
    # translate the origo to the bottom left corner
    offset = tf.constant([[1., 1., 1., 1., 0., 0.]], dtype=tf.float32)
    return walls + offset


def force_grid_windows(walls):
    start_x = walls[..., 0]
    start_y = walls[..., 1]
    end_x = walls[..., 2]
    end_y = walls[..., 3]

    x_mean = (start_x + end_x) / 2.
    y_mean = (start_y + end_y) / 2.
    x_distance = tf.abs(start_x - end_x)
    y_distance = tf.abs(start_y - end_y)

    start_x_grid = tf.where(x_distance > y_distance, start_x, x_mean)
    start_y_grid = tf.where(x_distance > y_distance, y_mean, start_y)
    end_x_grid = tf.where(x_distance > y_distance, end_x, x_mean)
    end_y_grid = tf.where(x_distance > y_distance, y_mean, end_y)

    start_grid = tf.stack([start_x_grid, start_y_grid], axis=-1)
    end_grid = tf.stack([end_x_grid, end_y_grid], axis=-1)

    return tf.concat([start_grid, end_grid, walls[..., 4:]], axis=-1)


def force_grid_absolute(walls, delta):
    start = walls[..., :2].numpy()
    end = walls[..., 2:4].numpy()
    pl = len(start)
    points = np.concatenate([start, end], axis=0)

    xs = np.concatenate([start[..., 0], end[..., 0]], axis=0)
    ys = np.concatenate([start[..., 1], end[..., 1]], axis=0)
    xclusters, xclusterv = cluster(xs, delta)
    yclusters, yclusterv = cluster(ys, delta)

    for i, cv in enumerate(xclusterv):
        for pidx in xclusters[i]:
            points[pidx][0] = cv

    for i, cv in enumerate(yclusterv):
        for pidx in yclusters[i]:
            points[pidx][1] = cv

    start, end = points[:pl], points[pl:]

    return tf.concat([start, end, walls[..., 4:]], axis=-1)


def cluster(values, delta):
    args = np.argsort(values)

    clusters = [
        [args[0]],
    ]
    for pidx in args[1:]:
        last_cluster = clusters[-1]
        cluster_first = values[last_cluster[0]]
        distance_from_first = np.abs(values[pidx] - cluster_first)
        if distance_from_first < delta:
            last_cluster.append(pidx)
        else:
            clusters.append([pidx])

    cluster_values = [np.mean([values[i] for i in cs]) for cs in clusters]
    return clusters, cluster_values


def get_absolute_position(walls, nrows, ncols, window_size):
    walls = translate_origo(walls)

    coords = walls[..., :4]
    width = walls[..., 4]

    # [0,2] -> [0,1]
    coords = coords / 2.
    # [0,1] -> [0,window_size]
    coords = coords * window_size
    # from flat to matrix
    coords = tf.reshape(coords, shape=(nrows, ncols, *coords.shape[1:]))

    row_offsets = tf.range(0., nrows) * window_size
    row_offsets = tf.reverse(row_offsets, axis=[-1])
    col_offsets = tf.range(0., ncols) * window_size

    row_offsets = tf.repeat(row_offsets[..., tf.newaxis], ncols, axis=-1)
    col_offsets = tf.repeat(col_offsets[tf.newaxis, ...], nrows, axis=0)

    offsets = tf.stack([col_offsets, row_offsets], axis=-1)
    offsets = tf.concat([offsets, offsets], axis=-1)

    coords = coords + offsets[:, :, tf.newaxis, :]
    coords = tf.reshape(coords, shape=(nrows*ncols, *coords.shape[2:]))

    coords = coords / (nrows * window_size)
    coords = coords * 2.
    coords = coords - 1.

    width = width / nrows

    return tf.concat([coords, width[..., tf.newaxis], walls[..., -1][..., tf.newaxis]], axis=-1)


def filter_not_visible(walls):
    walls = walls.numpy()
    walls = [w for w in walls if w[-1] > 0.]
    return tf.constant(walls, dtype=tf.float32)
