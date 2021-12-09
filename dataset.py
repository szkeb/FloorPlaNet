import tensorflow as tf
from generator import generate_floorplan
from diffrend import render_walls


class FloorPlanSequence(tf.keras.utils.Sequence):
    def __init__(self, img_size, n_walls, full_size, batch_size):
        self.img_size = img_size
        self.n_walls = n_walls
        self.full_size = full_size
        self.batch_size = batch_size

    def __len__(self):
        return self.full_size

    def __getitem__(self, idx):
        fp = generate_floorplan((self.batch_size, self.n_walls))
        imgs = render_walls(fp, self.img_size, 0.)
        imgs = tf.clip_by_value(imgs, 0., 1.)
        return imgs, fp


if __name__ == "__main__":
    from utils import display

    fps = FloorPlanSequence(256, 4, 64, 32)
    images, params = fps[123]

    for p in params[:3]:
        print(p)

    display(images[:3], show=True)
