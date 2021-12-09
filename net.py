import tensorflow as tf
from tensorflow.keras.models import Sequential
import layers
import numpy as np
from hungarian import hungarian_method
from generator import sort_coords, distance, sort_walls


class FloorPlaNet(tf.keras.Model):
    def __init__(self, img_size, parameter_shape, initial_blur, loss_weights):
        super(FloorPlaNet, self).__init__()
        self.img_size = img_size
        self.n_walls = parameter_shape[0]
        self.n_parameters = parameter_shape[1]

        self.mse_weight = loss_weights[0]
        self.prenet_weight = loss_weights[1]
        self.param_weight = loss_weights[2]

        self.encoder = self.get_encoder()
        self.prenet = self.get_prenet(input_shape=self.encoder.output_shape[1:])
        self.parametrizer = self.get_parametrizer(input_shape=self.encoder.output_shape[1:])
        self.renderer = layers.Renderer(self.img_size, initial_blur, name="Renderer")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.img_mse_tracker = tf.keras.metrics.Mean(name='img_MSE')
        self.param_loss_tracker = tf.keras.metrics.Mean(name='param_loss')
        self.prenet_loss_tracker = tf.keras.metrics.Mean(name='prenet_loss')

    @tf.function
    def custom_loss(self, y_true, y_pred):
        img_true, params_true = y_true
        img_pred, params_pred, n_walls = y_pred

        mse = self.img_loss(img_true, img_pred)
        param_loss = self.param_loss(params_true, params_pred)

        nw_true = tf.reduce_sum(params_true[..., -1], axis=-1)
        n_loss = tf.keras.losses.sparse_categorical_crossentropy(nw_true, n_walls)

        loss = mse * self.mse_weight + n_loss * self.prenet_weight + param_loss * self.param_weight

        return loss, mse, param_loss, n_loss

    @tf.function
    def img_loss(self, img_true, img_pred):
        mse = tf.square(img_true - img_pred)
        mse = tf.reduce_mean(mse, axis=[1, 2])

        return mse

    @tf.function
    def param_loss(self, params_true, params_pred):
        # [B, NW, 6] -> [B, NW(true), NW(pred), 6]
        pred_mx = tf.repeat(params_pred[:, tf.newaxis, ...], self.n_walls, axis=1)
        true_mx = tf.repeat(params_true[:, :, tf.newaxis, ...], self.n_walls, axis=2)
        # [B, NW(true), NW(pred)]
        # vertical axis -> true, horizontal axis -> pred
        edges = self.distance_function(true_mx, pred_mx)
        # [B, NW(true), NW(pred)]
        # vertical axis -> true, horizontal axis -> pred
        minimal_pairing = hungarian_method(edges)

        # sparse matrix with losses, where the minimal pairing is True
        loss = tf.where(minimal_pairing, edges, 0.)
        # sum the losses per batch
        loss = tf.reduce_sum(loss, axis=[1, 2])
        return loss

    @tf.function
    def distance_function(self, params_true, params_pred):
        probs_true = params_true[..., -1]
        probs_pred = params_pred[..., -1]
        descriptor_true = params_true[..., :-1]
        descriptor_pred = params_pred[..., :-1]

        prob_distance = self.prob_distance_function(probs_true, probs_pred)
        param_distance = self.param_distance_function(descriptor_true, descriptor_pred)
        return 10. * prob_distance + param_distance*probs_true

    @staticmethod
    @tf.function
    def prob_distance_function(probs_true, probs_pred):
        return tf.square(probs_true - probs_pred)

    @tf.function
    def param_distance_function(self, params_true, params_pred):
        start_true = params_true[..., :2]
        start_pred = params_pred[..., :2]
        end_true = params_true[..., 2:4]
        end_pred = params_pred[..., 2:4]
        width_true = params_true[..., 4]
        width_pred = params_pred[..., 4]
        dif_true = end_true - start_true
        dif_pred = end_pred - start_pred
        center_true = (start_true + end_true) / 2.
        center_pred = (start_pred + end_pred) / 2.
        length_true = distance(dif_true)
        length_pred = distance(dif_pred)

        center_distance = tf.square(distance(center_true - center_pred))

        area_true = length_true * width_true
        area_pred = length_pred * width_pred
        area_distance = tf.square(area_true - area_pred)

        lw_ratio_true = self.side_ratio(width_true, length_true)
        lw_ratio_pred = self.side_ratio(width_pred, length_pred)
        lwr_distance = tf.square(lw_ratio_true - lw_ratio_pred)

        bb_area_true, bbx_true, bby_true = self.calculate_bounding_box_area(start_true, end_true, dif_true, width_true)
        bb_area_pred, bbx_pred, bby_pred = self.calculate_bounding_box_area(start_pred, end_pred, dif_pred, width_pred)
        area_ratio_true = tf.where(bb_area_true > 0.001, area_true / bb_area_true, 1.)
        area_ratio_pred = tf.where(bb_area_pred > 0.001, area_pred / bb_area_pred, 1.)
        rotation_distance = tf.square(area_ratio_true - area_ratio_pred)

        horizontalness_distance = tf.square(tf.abs(bbx_true - bbx_pred))
        verticalness_distance = tf.square(tf.abs(bby_true - bby_pred))

        param_distance = area_distance + center_distance + rotation_distance + horizontalness_distance + verticalness_distance + lwr_distance
        return param_distance

    @staticmethod
    @tf.function
    def calculate_bounding_box_area(start, end, dif, width):
        per = tf.concat([dif[..., 1][..., tf.newaxis], -dif[..., 0][..., tf.newaxis]], axis=-1)
        per_dist = distance(per)[..., tf.newaxis]
        per = tf.where(per_dist > 0., per / per_dist, per)
        per = per * width[..., tf.newaxis] / 2.

        corners = tf.stack([
            start + per,
            start - per,
            end + per,
            end - per
        ], axis=-2)

        right = tf.reduce_max(corners[..., 0], axis=-1)
        top = tf.reduce_max(corners[..., 1], axis=-1)
        center = (start + end) / 2.

        bb_width = tf.abs((right - center[..., 0]) * 2.)
        bb_height = tf.abs((top - center[..., 1]) * 2.)
        bb_area = bb_width * bb_height

        return bb_area, bb_width, bb_height

    @staticmethod
    @tf.function
    def side_ratio(a, b):
        smaller = tf.where(a <= b, a, b)
        bigger = tf.where(a > b, a, b)

        return tf.where(bigger > 0., smaller/bigger, 0.)

    @tf.function
    def train_step(self, data):
        img_original, params_original = data
        with tf.GradientTape() as tape:
            y_pred = self(img_original, training=True)
            losses = self.custom_loss(data, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(losses[0], trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.update_metrics(losses)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        img_original, params_original = data

        y_pred = self(img_original, training=False)
        losses = self.custom_loss(data, y_pred)

        self.update_metrics(losses)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, **kwargs):
        feature_map = self.encoder(inputs, **kwargs)

        n_walls_probs = self.prenet(feature_map, **kwargs)
        n_walls = tf.argmax(n_walls_probs, axis=-1)
        probs_mask = tf.sequence_mask(n_walls, self.n_walls)
        probs = tf.where(probs_mask, 1., 0.)[..., tf.newaxis]

        parameters = self.parametrizer(feature_map, **kwargs)
        parameters = tf.concat([parameters, probs], axis=-1)

        img = self.renderer(parameters, **kwargs)

        return img, parameters, n_walls_probs

    def get_encoder(self):
        return Sequential([
            layers.InputLayer(input_shape=(self.img_size, self.img_size, 1)),
            tf.keras.layers.Lambda(tf.image.grayscale_to_rgb),
            tf.keras.applications.efficientnet.EfficientNetB1(
                include_top=False, input_shape=(self.img_size, self.img_size, 3), pooling='avg'
            )
        ])

    def get_parametrizer(self, input_shape):
        return Sequential([
            layers.InputLayer(input_shape=input_shape, name="param_input"),
            layers.Flatten(name="flatten"),
            layers.Dense(256, activation=layers.LeakyReLU(), name="hidden1"),
            layers.Dropout(0.1, name='dropout'),
            layers.Dense(256, activation=layers.LeakyReLU(), name="hidden2"),
            layers.Dense(self.n_walls*self.n_parameters, name="parameters"),
            layers.Reshape(target_shape=(self.n_walls, self.n_parameters),
                           input_shape=(self.n_walls * self.n_parameters,),
                           name="reshape"),
            layers.Normalizer(name="normalizer"),
        ], name="Parametrizer")

    def get_prenet(self, input_shape):
        return Sequential([
            layers.InputLayer(input_shape=input_shape, name="prenet_input"),
            layers.Flatten(name="prenet_flatten"),
            layers.Dense(128, activation=layers.LeakyReLU(), name="prenet_hidden"),
            layers.Dropout(0.1, name='prenet_dropout'),
            layers.Dense(self.n_walls+1, activation=layers.Softmax(), name="probabilites"),
        ], name="Prenet")

    def update_metrics(self, losses):
        for loss, metric in zip(losses, self.metrics):
            metric.update_state(loss)

    @property
    def metrics(self):
        return [self.loss_tracker,
                self.img_mse_tracker,
                self.param_loss_tracker,
                self.prenet_loss_tracker]
