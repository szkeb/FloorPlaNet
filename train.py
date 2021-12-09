import tensorflow as tf
import os
from net import FloorPlaNet
import dataset
import train_utils
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


IMG_SIZE = 256
N_WALLS = 4
PARAM_SHAPE = (N_WALLS, 5)
PEAK_DIR = "peaks"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs of training", type=int)
    parser.add_argument("--blur", help="Initial value of blur on rendered images", type=float)
    parser.add_argument("--lr", help="Initial learning rate", type=float)
    parser.add_argument("--batch", help="Batch size", type=int)
    parser.add_argument("--spe", help="Steps per epoch", type=int)
    parser.add_argument("--msew", help="Weight of the MSE loss component in the loss function", type=float)
    parser.add_argument("--prenetw", help="Weight of the 'prenet' loss component in the loss function", type=float)
    parser.add_argument("--paramw", help="Weight of the supervised loss component in the loss function", type=float)
    parser.add_argument('--load', help='Load model', action='store_true')
    parser.add_argument('--path', action='store', type=str, help='Path of the model/saved weights')

    args = parser.parse_args()

    loss_weights = (args.msew, args.prenetw, args.paramw)
    return args.epochs, args.blur, args.lr, args.batch, args.spe, args.load, loss_weights, args.path


if __name__ == "__main__":
    EPOCHS, INIT_BLUR, LR, BATCH_S, STEPS_PER_EPOCH, LOAD_MODEL, LOSS_WEIGHTS, WEIGHT_SAVE_PATH = parse_args()
    DS_SIZE = BATCH_S * STEPS_PER_EPOCH

    net = FloorPlaNet(IMG_SIZE, PARAM_SHAPE, INIT_BLUR, LOSS_WEIGHTS)
    if LOAD_MODEL:
        net.load_weights(WEIGHT_SAVE_PATH)
        print('Model loaded from: ', WEIGHT_SAVE_PATH)
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR))
    tf.keras.backend.set_value(net.optimizer.learning_rate, LR)
    print(net.optimizer.learning_rate)

    net.encoder.summary()
    net.parametrizer.summary()
    net.prenet.summary()

    ds = dataset.FloorPlanSequence(IMG_SIZE, N_WALLS, DS_SIZE, BATCH_S)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(WEIGHT_SAVE_PATH, monitor="loss",
                                                    save_best_only=True, save_weights_only=False),
                 train_utils.PeakCallback(net, IMG_SIZE, N_WALLS, PEAK_DIR),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9, patience=3, min_delta=0.001)]

    history = net.fit(ds,
                      epochs=EPOCHS,
                      batch_size=BATCH_S,
                      steps_per_epoch=STEPS_PER_EPOCH,
                      callbacks=callbacks)

    train_utils.save_history(history)

    train_utils.examine_results(net, 32)
