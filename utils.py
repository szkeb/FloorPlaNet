import matplotlib.pyplot as plt


def display(images, rows=1, cols=None, size=8, save_to=None, show=False):
    if cols is None:
        cols = len(images)

    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for i, img in enumerate(images):
        ax.ravel()[i].imshow(img, cmap="gray")
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    k = size
    h = len(images[0])
    w = len(images[0][0])
    ratio = w/h
    figure.set_size_inches(cols*k*ratio, rows*k)

    if show:
        plt.show()

    if save_to is not None:
        figure.savefig(save_to)
