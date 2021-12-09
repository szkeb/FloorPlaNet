import numpy as np


def create_grid(img, window_size=256):
    H = img.shape[-3]
    W = img.shape[-2]

    now_vertical = H//window_size + 1
    now_horizontal = W//window_size + 1
    pad_vertical = window_size * now_vertical - H
    pad_horizontal = window_size * now_horizontal - W

    padded_img = np.pad(img, [(0, pad_vertical), (0, pad_horizontal), (0, 0)], mode='constant', constant_values=0.)

    windows = []
    for yi in range(now_vertical):
        y = yi * window_size
        row = []
        for xi in range(now_horizontal):
            x = xi * window_size
            row.append(padded_img[y:y+window_size, x:x+window_size, :])
        windows.append(row)

    windows = np.array(windows)
    return windows


def glue_img(windows):
    for yi, y_block in enumerate(windows):
        for xi, x_block in enumerate(y_block):
            if xi == 0:
                row = x_block
            else:
                row = np.concatenate([row, x_block], axis=1)
        if yi == 0:
            img = row
        else:
            img = np.concatenate([img, row], axis=0)

    return img
