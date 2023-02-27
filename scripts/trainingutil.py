import datetime
import os
import random
import timeit
from math import ceil

import cv2
import numpy as np
import tensorflow as tf
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import rotate

from scripts.util import quantile_normalize, normalize_image, log_msg


class ImageCache:
    """"
    Cache object for caching and loading images in memory.

    Parameters
    ----------
    cache_size : int
        The number of images to allocate space for
    channels : int
        Number of channels in the images stored

    Attributes
    ----------
    cache : np.array
        The matrix containing the images
    cache_map : dict
        A dictionary mapping file paths to images in the cache
    channels : int
        The number of channels in the images stored
    """

    def __init__(self, cache_size, channels):
        self.cache = np.zeros((cache_size, 512, 512, channels))
        self.cache_map = {}
        self.channels = channels

    def load_image(self, path):
        """
        Loads image at path from memory if cached, otherwise loads from storage and caches it.
        Always returns a copy of the image, such that it can be safely manipulated.
        Parameters
        ----------
        path : str
            Path of the image
        Returns
        -------
        np.array
            the requested image
        """
        if path in self.cache_map:
            return np.copy(self.cache[self.cache_map[path]])
        else:
            new_idx = len(self.cache_map)
            self.cache_map[path] = new_idx
            img = tifffile.imread(path).astype('float32')

            if len(img.shape) == 3:  # if we load ZXY, move to XYZ
                img = np.moveaxis(img, 0, 2)
            elif len(img.shape) == 2:  # add the implicit Z channel, eve if it is 1
                img = img.reshape((512, 512, 1))
            else:  # unexpected input dims
                raise ValueError("Image at path is not 2D or 3D")

            if img.shape[0:2] != (512, 512):
                for i in range(self.channels):
                    img[:, :, i] = cv2.resize(img[:, :, i], (512, 512), interpolation=cv2.INTER_CUBIC)

            self.cache[new_idx] = img
            return np.copy(img)


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    m = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, m, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def load_dataset(fol):
    """
    Gets the category directories of a dataset and returns it
    along with the number of files contained in those directories.
    Used for categorized datasets.
    ----------
    dir : str
        Path of the dataset
    Returns
    -------
    np.array
        array containing dataset category directories
    int
        size of the dataset
    """
    dataset = np.array(os.listdir(fol))
    dataset_size = sum([len(files) for r, d, files in os.walk(fol)])
    return dataset, dataset_size


class ImageStackGenerator(tf.keras.utils.Sequence):
    """
    Data generator for TensorFlow model. Takes directories of categorized training data in sub-directories and samples a
    single image from each sub directory.
    Parameters
    ----------
    input_dir : str
        directory containing input data
    label_dir : str
        directory containing label data
    batch_size : int
        batch size for training steps
    augment : int
        0 = no augmentation, 1 = rotations, 2 = elastic deformation + rotation
    """

    def __init__(self, input_dir, label_dir, batch_size=8, augment_level=0):
        # label may not exist for every input, but every label has an input
        training_groups, dataset_size = load_dataset(label_dir)
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.training_groups = training_groups
        self.batch_size = batch_size
        self.augment = augment_level
        self.input_cache = ImageCache(dataset_size, 5)
        self.label_cache = ImageCache(dataset_size, 1)

    def __len__(self):
        return ceil(len(self.training_groups) / self.batch_size)

    def __getitem__(self, item):
        group_folders = self.training_groups.take(range(item * self.batch_size, (item + 1) * self.batch_size),
                                                  mode="wrap")  # get file names
        batch_x = np.empty((self.batch_size, 512, 512, 5))
        batch_y = np.empty((self.batch_size, 512, 512, 1))

        for _i, folder in enumerate(group_folders):

            # find a label file, since input may not have labels for every channel
            img_file = random.choice(os.listdir(f"{self.label_dir}/{folder}"))

            img = self.input_cache.load_image(f"{self.input_dir}/{folder}/{img_file}")
            # normalize the raw image.
            for i in range(5):
                img[:, :, i] = quantile_normalize(img[:, :, i])

            label = self.label_cache.load_image(f"{self.label_dir}/{folder}/{img_file}")
            label = normalize_image(label)  # force label to be between 0 and 1

            if self.augment == 2:  # aug method 2

                rotation = random.choice([0, 90, 180, 270])  # rotation amount
                img = rotate(img, rotation, axes=(1, 0))
                label = rotate(label, rotation, axes=(1, 0))

                rand_int = random.randint(1, 142857)

                # elastic deformation params found using testing on a blank grid.
                for i in range(5):
                    img[:, :, i] = elastic_transform(img[:, :, i], 430, 20, 40,
                                                     random_state=np.random.RandomState(rand_int))
                label = elastic_transform(label[:, :, 0], 430, 20, 40, random_state=np.random.RandomState(rand_int))

                # old augment method
            elif self.augment == 1:  # aug method 1
                rotation = random.choice([0, 90, 180, 270])  # rotation amount
                img = rotate(img, rotation, axes=(1, 0))
                label = rotate(label, rotation, axes=(1, 0))

            batch_x[_i] = img
            batch_y[_i] = label.reshape((512, 512, 1))  # make Z / C dimension explicit.
        return batch_x, batch_y


# loss functions for ModelTrainer class.
bce_log = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
mae = tf.keras.losses.MeanAbsoluteError()

lr_v = 3e-4
beta1 = 0.9


def dice_loss(y_true, y_pred):
    """
    Dice loss calculated as 1 - Continuous Dice Coefficient
    Parameters
    ----------
    y_true
        Ground truth image(s)
    y_pred
        Predicted image(s)
    Returns
    -------
    float
        Dice loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


class ModelTrainer:
    """
    Helper class for training a TensorFlow model with Dice + TopK binary cross entropy loss. Datagen must be done with
    ImageStackGenerator class, to ensure a matching training data scheme.
    Parameters
    ----------
    datagenerator : ImageStackGenerator
        The data generator instance containing the training data
    batch_size : int
        Size of mini batch during training step
    model : tf.keras.model
        model to train
    k
        k value for TopK loss
    model_save_dir
        directory to save trained model in
    save_frequency
        epochs between each model save
    """

    def __init__(self, datagenerator: ImageStackGenerator, batch_size=8, model: tf.keras.Model = None, k=0.1,
                 model_save_dir="out",
                 save_frequency=10):
        self.datagen = datagenerator
        self.batch_size = batch_size
        self.model = model
        self.k = k
        self.model_save_dir = model_save_dir
        self.save_frequency = save_frequency
        self.optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    def load_model(self, path):
        """
        Load the model to be trained.
        Parameters
        ----------
        path
            path of model
        """
        self.model = tf.keras.models.load_model(path)

    def train(self, epochs=30):
        """
        Train the loaded model
        Parameters
        ----------
        epochs : int
            number of epochs to train for
        """
        if self.model is None:
            raise ValueError("Model not loaded. Must be loaded in constructor or using load_model method.")

        for e in range(1, epochs + 1):
            epoch_loss = 0
            start = timeit.default_timer()
            for input_images, label_images in self.datagen:
                loss = self.train_step(input_images, label_images)
                epoch_loss += loss

            stop = timeit.default_timer()
            remaining = datetime.timedelta(seconds=round(stop - start) * (epochs - e))
            log_msg(f"Epoch {e} loss: {epoch_loss}. Remaining: {remaining}")

            if e % self.save_frequency == 0:
                self.model.save(f"{self.model_save_dir}/model-e{e}.hdf5")
        # save last model if it is not divisible by save_frequency
        if epochs % self.save_frequency != 0:
            self.model.save(f"{self.model_save_dir}/model-e{epochs}.hdf5")

    @tf.function
    def train_step(self, input_images, label_images):
        """
        Training step for a single batch
        Parameters
        ----------
        input_images
            input images for segmentation
        label_images
            segmentation labels
        Returns
        -------
        float
            loss training step
        """
        with tf.GradientTape(persistent=True) as tape:
            images_generated = self.model(input_images, training=True)
            bce_loss = bce(label_images, images_generated)
            num_voxels = np.prod(bce_loss.shape.as_list())
            k = num_voxels * self.k
            top_bce, top_indices = tf.raw_ops.TopKV2(input=tf.reshape(bce_loss, (-1,)), k=k)
            loss_topk = tf.math.reduce_mean(top_bce)
            loss_dice = dice_loss(label_images, images_generated)
            loss = loss_topk + loss_dice

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        return loss


class EvalStackGenerator(tf.keras.utils.Sequence):
    """
    Data generator for TensorFlow model. Takes directories of categorized training data in sub-directories.
    Applies quantile normalization.
    Parameters
    ----------
    input_dir : str
        directory containing input data
    label_dir : str
        directory containing label data
    batch_size : int
        batch size for training steps
    """

    def __init__(self, input_dir, label_dir, batch_size=8):
        # label may not exist for every input, but every label has an input
        training_groups, dataset_size = load_dataset(label_dir)
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.len = dataset_size
        self.batch_size = batch_size
        self.input_cache = ImageCache(dataset_size, 5)
        self.label_cache = ImageCache(dataset_size, 1)
        # populate cache, since it will all be used
        for dir in training_groups:
            # use label dirs to determine which should be added. They should be symmetric
            for img_file in os.listdir(f"{self.label_dir}/{dir}"):
                self.label_cache.load_image(f"{self.label_dir}/{dir}/{img_file}")
                self.input_cache.load_image(f"{self.input_dir}/{dir}/{img_file}")
        # verify that all images have actually been loaded
        assert (len(self.input_cache.cache_map) == self.len)
        assert (len(self.label_cache.cache_map) == self.len)

    def __len__(self):
        return ceil(self.len / self.batch_size)

    def __getitem__(self, item):
        raw_input = self.input_cache.cache.take(range(item * self.batch_size, (item + 1) * self.batch_size),
                                                mode="wrap", axis=0)
        raw_label = self.label_cache.cache.take(range(item * self.batch_size, (item + 1) * self.batch_size),
                                                mode="wrap", axis=0)
        batch_x = np.empty((self.batch_size, 512, 512, 5))
        batch_y = np.empty((self.batch_size, 512, 512, 1))

        for i in range(self.batch_size):
            for j in range(5):
                raw_input[i, :, :, j] = quantile_normalize(raw_input[i, :, :, j])

            raw_label[i] = normalize_image(raw_label[i])  # force label to be between 0 and 1

            batch_x[i] = raw_input[i]
            batch_y[i] = raw_label[i]
        return batch_x, batch_y
