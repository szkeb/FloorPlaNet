import cv2
from preprocess import *
import matplotlib.pyplot as plt


def find_doors(original):
    clean = CCC(original, 4, 4)
    arcs, _ = separate_straight_lines(clean, 30, min_line_length=40, max_line_gap=5)

    arcs_d = morphology.binary_dilation(arcs, morphology.square(5))
    arcs_dc = CCC(arcs_d, 10, 10)
    arcs_dcd = morphology.binary_dilation(arcs_dc, morphology.square(3))
    arcs_dcdc = CCC(arcs_dcd, 50, 50)

    arcs_filtered = filter_arcs(arcs_dcdc, 0.05, 0.35)

    doors = get_doors(arcs_filtered)
    doors = [(d[0]+d[-2], d[1]+d[-1]) for d in doors]

    return doors


if __name__ == "__main__":
    original = read_and_invert("floorplan.png")

    images = [original, thick, medium, thin] = separate_elements(original)
    images = [invert(i) for i in images]

    doors = get_doors(thin)

    print(doors)
