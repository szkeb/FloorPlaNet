from net import FloorPlaNet
import argparse
from train import IMG_SIZE, PARAM_SHAPE
from preprocess import read_and_invert, separate_elements
import numpy as np
from doors import find_doors
from grid import create_grid
import tensorflow as tf
from postprocess  import postprocess_params
from diffrend import render_walls
from utils import display


def get_model(path):
    net = FloorPlaNet(IMG_SIZE, PARAM_SHAPE, 0., [0., 0., 0., 0.])
    print('Loading model weights from: ', path)
    net.load_weights(path)
    return net


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mina", help="Minimum area of walls", type=float)
    parser.add_argument("--da", help="Sensitivity of the snapping of walls to grid", type=float)
    parser.add_argument("--md", help="Sensitivity of the wall-merging", type=float)
    parser.add_argument("--dd", help="Sensitivity of the door snapping", type=float)
    parser.add_argument('--path', action='store', type=str, help='Path of the model/saved weights')
    parser.add_argument('--out', action='store', type=str, help='Output filename')
    parser.add_argument('--fp', action='store', type=str, help='Input filepath')

    args = parser.parse_args()

    return args.mina, args.da, args.md, args.dd, args.path, args.out, args.fp


if __name__ == "__main__":
    WINDOW_SIZE = 256
    MIN_AREA, DELTA_ABSOLUTE, MERGE_DELTA, DOOR_DELTA, PATH, OUT, FPNAME = parse_args()

    original = read_and_invert(FPNAME)
    [original, thick, medium, thin] = separate_elements(original)
    doors_pixels = find_doors(original)
    doors_pixels = np.asarray([[d / original.shape[0] for d in door] for door in doors_pixels])
    doors = np.asarray([[d[0], 1.-d[1]] for d in doors_pixels])
    doors = (doors - 0.5) * 2.

    input = thick[..., np.newaxis]
    input = create_grid(input, WINDOW_SIZE)
    rows, cols = input.shape[:2]
    input = tf.constant(input, dtype=tf.float32)
    input = tf.reshape(input, shape=(rows*cols, WINDOW_SIZE, WINDOW_SIZE, 1))

    net = tf.keras.models.load_model(PATH)
    # or
    # net = get_model(PATH)
    output_windows_raw, params, _ = net(input, training=False)

    processed_params, doors = postprocess_params(params, doors,
                                                 min_area=MIN_AREA, delta_absolute=DELTA_ABSOLUTE,
                                                 merge_delta=MERGE_DELTA, door_delta=DOOR_DELTA,
                                                 nrows=rows, ncols=cols, window_size=WINDOW_SIZE)

    rendered_imgs = [render_walls(params[tf.newaxis, ...], 1024)[0, ..., 0] for params in processed_params]

    display([thick, *rendered_imgs], save_to=f"{OUT}.png")

    np.savetxt(f"{OUT}.txt", processed_params[-1])
    np.savetxt(f"{OUT}_doors.txt", doors)