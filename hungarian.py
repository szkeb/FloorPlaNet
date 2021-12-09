import tensorflow as tf
import numpy as np


def hungarian_method(edge_mx):
    edge_mx = row_reduction(edge_mx)
    edge_mx = column_reduction(edge_mx)
    assignment = allocate(edge_mx)
    rows, cols = wiki_lines(edge_mx, assignment)

    # while no assignment is possible
    # tf.where makes sure not to do it more times than it is necessary
    for i in range(edge_mx.shape[-1] * 2):
        edge_mx = tf.where(check_lines(edge_mx, rows, cols),
                           edge_mx,
                           shift_zeros(edge_mx, rows, cols))
        assignment = allocate(edge_mx)
        rows, cols = wiki_lines(edge_mx, assignment)

    assignment = allocate(edge_mx)
    return assignment


def row_reduction(edge_mx):
    mins = tf.reduce_min(edge_mx, axis=-1)
    mins = mins[..., tf.newaxis]
    edge_mx = edge_mx - mins
    return edge_mx


def column_reduction(edge_mx):
    mins = tf.reduce_min(edge_mx, axis=-2)
    mins = mins[:, tf.newaxis, ...]
    edge_mx = edge_mx - mins

    return edge_mx


def create_lines(edge_mx):
    batch_dim, rank, _ = edge_mx.shape
    mx = edge_mx
    out_mx = tf.cast(tf.zeros(shape=(batch_dim, rank * 2, rank)), tf.bool)

    for i in range(rank):
        transposed = tf.transpose(mx, perm=[0, 2, 1])
        double = tf.concat([mx, transposed], axis=1)

        zeros = tf.where(double == 0., 1., 0.)
        n_zeros = tf.reduce_sum(zeros, axis=[-1])

        max_part = tf.argmax(n_zeros, axis=-1, output_type=tf.int32)
        max_num = tf.reduce_max(n_zeros, axis=-1)

        mask = assign_mask(rank * 2, rank, max_part)
        out_mx = tf.where(max_num[:, tf.newaxis, tf.newaxis] != 0., tf.where(mask, True, out_mx), out_mx)

        mx = tf.where(max_part[:, tf.newaxis, tf.newaxis] >= rank, tf.transpose(mx, perm=[0, 2, 1]), mx)
        max_part_mod = max_part % rank
        mask = assign_mask(rank, rank, max_part_mod)
        mx = tf.where(mask, mx + 1., mx)
        mx = tf.where(max_part[:, tf.newaxis, tf.newaxis] > rank, tf.transpose(mx, perm=[0, 2, 1]), mx)

    rows, cols = tf.split(out_mx, [rank, rank], axis=-2)
    cols = tf.transpose(cols, perm=(0, 2, 1))

    return rows, cols


def wiki_lines(edge_mx, assignment):
    rank = edge_mx.shape[-1]

    markings = mark_unassigned(edge_mx, assignment)

    new_cols = mark_columns(edge_mx, markings)
    new_rows = mark_rows(assignment, new_cols)
    markings = tf.logical_or(markings, new_cols)
    markings = tf.logical_or(markings, new_rows)

    for i in range(rank):
        new_cols = mark_columns(edge_mx, new_rows)
        new_rows = mark_rows(assignment, new_cols)
        markings = tf.logical_or(markings, new_cols)
        markings = tf.logical_or(markings, new_rows)

    unmarked_rows = tf.reduce_all(markings, axis=[-1])
    unmarked_rows = tf.logical_not(unmarked_rows)
    unmarked_rows = tf.repeat(unmarked_rows[..., tf.newaxis], rank, axis=-1)

    marked_cols = tf.reduce_all(markings, axis=[-2])
    marked_cols = tf.repeat(marked_cols[:, tf.newaxis, ...], rank, axis=-2)

    return unmarked_rows, marked_cols


def mark_unassigned(edge_mx, assignment):
    rank = edge_mx.shape[-1]
    assigned_rows = tf.reduce_any(assignment, axis=[-1])
    unassigned_rows = tf.logical_not(assigned_rows)
    unassigned_rows = tf.repeat(unassigned_rows[..., tf.newaxis], rank, axis=-1)
    return unassigned_rows


def mark_columns(edge_mx, new_rows):
    rank = edge_mx.shape[-1]
    cells = tf.logical_and(edge_mx == 0., new_rows)
    columns = tf.reduce_any(cells, axis=[-2])
    columns = tf.repeat(columns[:, tf.newaxis, ...], rank, axis=-2)

    return columns


def mark_rows(assignment, new_cols):
    rank = assignment.shape[-1]
    cells = tf.logical_and(assignment, new_cols)
    rows = tf.reduce_any(cells, axis=[-1])
    rows = tf.repeat(rows[..., tf.newaxis], rank, axis=-1)

    return rows


def assign_mask(n_rows, n_columns, idx):
    before = tf.sequence_mask(idx, n_rows)
    after = tf.sequence_mask(n_rows - idx - 1, n_rows)
    after = tf.reverse(after, axis=[-1])

    mask = tf.logical_or(before, after)
    mask = tf.logical_not(mask)
    mask = tf.repeat(mask[..., tf.newaxis], n_columns, axis=-1)

    return mask


def check_lines(edge_mx, rows, cols):
    rows = tf.where(rows[..., 0], 1, 0)
    n_rows = tf.reduce_sum(rows, axis=[-1])

    cols = tf.transpose(cols, perm=[0, 2, 1])
    cols = tf.where(cols[..., 0], 1, 0)
    n_cols = tf.reduce_sum(cols, axis=[-1])

    n_lines = n_rows + n_cols
    n_mx = edge_mx.shape[-1]

    return tf.equal(n_lines, n_mx)[:, tf.newaxis, tf.newaxis]


def shift_zeros(edge_mx, rows, cols):
    cover = tf.logical_or(rows, cols)
    uncovered = tf.where(cover, 1000000., edge_mx)
    min_uncovered = tf.reduce_min(uncovered, axis=[-2, -1])

    edge_mx = add_to_cross(edge_mx, min_uncovered, rows, cols)
    edge_mx = subtract_from_uncovered(edge_mx, min_uncovered, cover)

    return edge_mx


def add_to_cross(edge_mx, min_value, rows, cols):
    cross = tf.logical_and(rows, cols)
    edge_mx = tf.where(cross, edge_mx + min_value[:, tf.newaxis, tf.newaxis], edge_mx)
    return edge_mx


def subtract_from_uncovered(edge_mx, min_value, cover):
    edge_mx = tf.where(cover, edge_mx, edge_mx - min_value[:, tf.newaxis, tf.newaxis])
    return edge_mx


def allocate(edge_mx):
    rank = edge_mx.shape[-1]
    mx = edge_mx
    out_mx = tf.cast(tf.zeros_like(edge_mx), tf.bool)

    for i in range(rank):
        transposed = tf.transpose(mx, perm=[0, 2, 1])
        double = tf.concat([mx, transposed], axis=1)
        zeros = tf.where(double == 0., 1., 0.)
        # [B, H] : how many zeros in each row
        n_zeros = tf.reduce_sum(zeros, axis=[-1])
        # [B, H] : filter out (cheat) the rows where no zeros are
        n_zeros = tf.where(n_zeros == 0., 100., n_zeros)

        row_idx = tf.argmin(n_zeros, axis=-1, output_type=tf.int32)

        # which zero from row/column based on the other axis
        # the goal is to choose the one that eliminates the least zeros
        n_zeros = tf.where(double == 0.,
                           tf.reduce_sum(zeros, axis=[-1])[..., tf.newaxis],
                           1000)
        n_zeros_rows = n_zeros[:, :rank]
        n_zeros_columns = n_zeros[:, rank:]
        n_zeros = tf.concat([n_zeros_rows + transpose_mx(n_zeros_columns),
                            n_zeros_columns + transpose_mx(n_zeros_rows)],
                            axis=1)
        col_idx = tf.argmin(n_zeros, axis=-1, output_type=tf.int32)

        before = tf.sequence_mask(col_idx, rank)
        after = tf.sequence_mask(rank - col_idx - 1, rank)
        after = tf.reverse(after, axis=[-1])
        mask_col = tf.logical_or(before, after)
        mask_col = tf.logical_not(mask_col)

        # [B, H, W] : full T\F matrix, where only the chosen row is True
        mask_row = assign_mask(rank * 2, rank, row_idx)

        # [B, H, W] : full T\F matrix, where only one cell is True
        mask = tf.logical_and(mask_row, mask_col)
        # [B, H, W] : check if it is really a zero element.
        # This step is necessary because if there is no possible assignment, argmins will choose the first option.
        mask = tf.logical_and(mask, double == 0.)

        mask_original = mask[:, :rank]
        mask_transposed = mask[:, rank:, :]
        mask_transposed = tf.transpose(mask_transposed, perm=(0, 2, 1))
        mask = tf.logical_or(mask_original, mask_transposed)

        # [B, H] : which column
        column_mask = tf.reduce_any(mask, axis=[-2])
        # [B, W, H] : fill the column with True
        column_mask = tf.repeat(column_mask[:, tf.newaxis, ...], rank, axis=-2)

        row_mask = tf.reduce_any(mask, axis=[-1])
        row_mask = tf.repeat(row_mask[..., tf.newaxis], rank, axis=-1)
        # [B, W, H] : merging column, and row mask
        update_mask = tf.logical_or(column_mask, row_mask)

        out_mx = tf.where(mask, True, out_mx)
        mx = tf.where(update_mask, 100., mx)

    return out_mx


def transpose_mx(mx):
    return tf.transpose(mx, perm=[0, 2, 1])