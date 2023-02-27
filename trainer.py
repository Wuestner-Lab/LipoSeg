#!/usr/bin/env python
import argparse
import os
import os.path

import tensorflow as tf

from scripts.resunet import res_unet
from scripts.trainingutil import ImageStackGenerator, ModelTrainer
from scripts.util import positive_int, restricted_float


def main():
    # Hide TF logging.
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Train a model on a specified dataset. "
                                                 "Model must take 512x512x5 input and 512x512x1 labels. "
                                                 "It is recommended to use a pretrained model. "
                                                 "Training data is sorted into directories consisting of groups of "
                                                 "similar data (for example segmentations from the same tomogram). "
                                                 "Each epoch, one image is sampled from each category, to prevent "
                                                 "overrepresentation of 1 group.")
    parser.add_argument("-m", "--model", help="the model to train", type=str, required=True)
    parser.add_argument("-o", "--out", help="output directory", default="out", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--input", help="directory of input images", type=str, required=True)
    parser.add_argument("--labels", help="directory of label images", type=str, required=True)
    parser.add_argument("-e", "--epochs", help="'epochs' to run model for", type=positive_int, default=50)
    parser.add_argument("-s", "--save_frequency", help="epochs between each model save", type=positive_int, default=10)
    parser.add_argument("-a", "--augmentation",
                        help="Level of augmentation to use. "
                             "0 = none, 1 = rotations, 2 = rotations and elastic deformation",
                        type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("-k", help="k value for TopK loss", default=0.1, type=restricted_float)
    parser.add_argument("--batch_size", help="batch size for training", default=8, type=positive_int)
    parser.add_argument("--shutdown", help="shutdown after completion", action="store_true")

    args = parser.parse_args()
    input_dir = args.input
    label_dir = args.labels
    if args.model == "resunet":
        model = res_unet(512,[32,64,128,256,512,1024],(3,3),5,1)
    else:
        model = tf.keras.models.load_model(args.model)

    datagen = ImageStackGenerator(input_dir, label_dir, batch_size=args.batch_size, augment_level=args.augmentation)
    os.makedirs(args.out, exist_ok=True)
    trainer = ModelTrainer(datagen, batch_size=args.batch_size, model=model, k=args.k,
                           save_frequency=args.save_frequency, model_save_dir=args.out)
    trainer.train(epochs=args.epochs)
    if args.shutdown:
        os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
