import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology


def norm_to_stand(img):
    if np.max(img) == 1.:
        return np.uint8(img*255)
    return np.uint8(img)


def stand_to_norm(img):
    if np.max(img) == 255:
        return img/255.
    return img


def invert(img):
    img = 1. - img
    return img


def read(path):
    img = plt.imread(path)
    img = img[:,:,0]
    return img


def read_and_invert(path):
    img = read(path)
    img = invert(img)
    return img


def to_binary(img):
    return np.uint8(np.where(img > 0, 255, 0))


def CCC(img, w_min=10, h_min=10):
    """
    Cleaning connected components based on width and height
    """
    img = norm_to_stand(img)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    nLabels = np.zeros_like(labels)

    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        if w > w_min and h > h_min:
            current = np.where(labels == i, i, 0)
            nLabels += current

    return to_binary(nLabels)


def separate_straight_lines(img, rho=38, min_line_length=5, max_line_gap=10):
    img = norm_to_stand(img)
    only_lines = np.zeros(shape=(img.shape[0], img.shape[1], 3))
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, rho, minLineLength=min_line_length, maxLineGap=max_line_gap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(only_lines, (x1, y1), (x2, y2), (255, 255, 255), 3)
    only_lines = only_lines[:, :, 0]
    without_lines = np.clip(img - only_lines, 0, 255)

    return without_lines, only_lines


def show_connected_components(img):
    img = norm_to_stand(img)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    res = np.repeat(img[..., np.newaxis], 3, axis=-1)
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return res


def show_connected_components_boundaries(img):
    img = norm_to_stand(img)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    res = np.zeros_like(img)
    res = np.repeat(res[..., np.newaxis], 3, axis=-1)
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return res


def filter_arcs(img, min_area, max_area, epsilon=0.01):
    img = norm_to_stand(img)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    nLabels = np.zeros_like(labels)
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        full = w*h
        if full * max_area * (1+epsilon) > area > full*min_area*(1-epsilon):
            current = np.where(labels == i, i, 0)
            nLabels += current

    return to_binary(nLabels)


def get_thick_components(img, n=3):
    img = stand_to_norm(img)
    for i in range(n):
        img = morphology.binary_erosion(img, morphology.square(3))
    for i in range(n+1):
        img = morphology.binary_dilation(img, morphology.square(3))
    return img


def subtract(from_sub, to_sub):
    remaining = from_sub - to_sub
    remaining = np.maximum(remaining, np.zeros_like(remaining))
    return remaining


def remove_thin_lines(img):
    horizontal = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    vertical = np.transpose(horizontal)

    img = morphology.binary_opening(img, horizontal)
    img = morphology.binary_opening(img, vertical)
    return img


def separate_elements(original):
    thick = get_thick_components(original)
    medium_and_thin = subtract(original, thick)
    medium = remove_thin_lines(medium_and_thin)
    thin = subtract(medium_and_thin, medium)

    return [original, thick, medium, thin]


def get_doors(img):
    img = norm_to_stand(img)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)

    doors = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        bounding_box = img[y:y+h, x:x+w]
        (horizontal, vertical) = find_orientation(bounding_box)
        doors.append((x, y, w, h, horizontal, vertical))

    return doors


def find_orientation(img, epsilon=1.5):
    w, h, = img.shape[1], img.shape[0]
    half = int(h / 2)
    top = np.sum(img[:half])
    bottom = np.sum(img[half:])

    half = int(w / 2)
    left = np.sum(img[:, :half])
    right = np.sum(img[:, half:])

    if 0.8 < w/h < 1.2:
        if top > bottom:
            if left > right:
                return w, h
            else:
                return 0, h
        else:
            if left > right:
                return w, 0
            else:
                return 0, 0
    else:
        if w > h:
            if top < bottom:
                return w/2, 0
            else:
                return w/2, h
        else:
            if left < right:
                return 0, h/2
            else:
                return w, h/2


def draw_doors(img, doors):
    img = show_connected_components(img)

    for (x, y, w, h, horizontal, vertical) in doors:
        img = cv2.circle(img, (int(x+horizontal), int(y+vertical)), 10, (0, 255, 0), cv2.FILLED)

    return img


def project_doors(img, last, doors):
    bounds = show_connected_components_boundaries(last)

    for (x, y, w, h, horizontal, vertical) in doors:
        bounds = cv2.circle(bounds, (int(x + horizontal), int(y + vertical)), 10, (0, 255, 0), cv2.FILLED)

    img = norm_to_stand(img)[..., np.newaxis]
    img = np.repeat(img, 3, axis=-1)
    img = np.where(np.sum(bounds, axis=-1)[..., np.newaxis] != 0, bounds, img)

    return img


if __name__ == "__main__":
    from utils import display

    original = read_and_invert("floorplan.png")

    images = [original, thick, medium, thin] = separate_elements(original)
    images = [invert(i) for i in images]

    display(images, save_to="separation.png", show=True)
