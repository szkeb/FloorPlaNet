import tensorflow as tf
from generator import generate_floorplan
from diffrend import render_walls
from utils import display
import os
import matplotlib.pyplot as plt
from datetime import datetime


class PeakCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, saves the output of the net, given the same image sequence every time.
    """
    def __init__(self, net, img_size, n_walls, out_dir):
        super().__init__()
        self.net = net
        self.img_size = img_size
        self.n_walls = n_walls
        self.output_dir = out_dir
        self.fix_dir = os.path.join(self.output_dir, 'fix')
        if not os.path.exists(self.fix_dir):
            os.mkdir(self.fix_dir)

        self.fixp = generate_floorplan((1, self.n_walls))
        self.fixi = tf.clip_by_value(render_walls(self.fixp, self.img_size, 0.), 0., 1.)

    def on_epoch_end(self, epoch, logs=None):
        fp = generate_floorplan((1, self.n_walls))
        input_img = render_walls(fp, self.img_size, 0.)
        output_img, output_params, nw = self.net(input_img, training=False)

        input_img = tf.clip_by_value(input_img, 0., 1.)
        output_img = tf.clip_by_value(output_img, 0., 1.)
        print('Original:\n', fp[0])
        print('Output:\n', output_params[0])
        display([input_img[0, :, :, 0], output_img[0, :, :, 0]], save_to=self.get_save_path(epoch+1))

        _, op, _ = self.net(self.fixi, training=False)
        fixo = render_walls(op, self.img_size, self.net.renderer.blur)
        display([self.fixi[0, :, :, 0], fixo[0, :, :, 0]], save_to=self.get_save_path(epoch+1, fix=True))

    def get_save_path(self, epoch, fix=False):
        dir = self.fix_dir if fix else self.output_dir
        return os.path.join(dir, f"peak_epoch#{epoch}.png")


def examine_results(net, n=4, path='results.png'):
    fp = generate_floorplan((n, net.n_walls))
    input_img = render_walls(fp, net.img_size, 0.)
    output_img, output_params, _ = net(input_img, training=False)

    input_img = tf.clip_by_value(input_img, 0., 1.)
    output_img = tf.clip_by_value(output_img, 0., 1.)

    imgs = []
    for original, pred in zip(input_img, output_img):
        imgs.append(original[:, :, 0])
        imgs.append(pred[:, :, 0])

    print(len(imgs))

    display(imgs, rows=n, cols=2, save_to=path)


def save_history(history, dir='lossplots'):
    losses = [k for k in history.history.keys() if k != 'lr']

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.tight_layout()

    for loss in losses:
        ax[0].plot(history.history[loss])
    ax[0].set_title('Losses')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(losses, loc='upper left')

    ax[1].plot(history.history['lr'])
    ax[1].set_title('LR')
    ax[1].set_ylabel('lr')
    ax[1].set_xlabel('epoch')

    now = datetime.now()
    current_time = now.strftime("%m%d_%Hh%Mm%Ss")
    path = os.path.join(dir, current_time)

    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.savefig(path)

