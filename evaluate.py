'''
Takes datagen and evaluates dice metric for each sample (for each model).
This can then be plotted with error bars using stddev

'''
import argparse
import os
import os.path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

import scripts.resunet
from scripts.trainingutil import EvalStackGenerator
from scripts.util import positive_int, restricted_float


def evaluate_model(model, data_generator: EvalStackGenerator, metrics, threshold=0.5):
    dice_scores = []
    for input, gt in data_generator:
        pred = model.predict(input)
        for i in range(data_generator.batch_size):
            gt_bi = (gt[i] > threshold).flatten()
            cmp_bi = (pred[i] > threshold).flatten()
            if "dice" in metrics:
                dice_scores.append(f1_score(gt_bi, cmp_bi, zero_division=1))
    return np.array(dice_scores)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Evaluate model(s) on a specified dataset using specified metric(s). "
                                                 "Evaluates on tensorflow model alone without post-processing")
    parser.add_argument("-m", "--model", help="path to model or directory containing models", type=str, required=True)
    parser.add_argument("-o", "--out", help="output directory", default="out", type=str)
    parser.add_argument("--input", help="directory of input images", type=str, required=True)
    parser.add_argument("--labels", help="directory of ground truth label images", type=str, required=True)
    parser.add_argument("--metrics", help="metrics to evaluate with", type=str, default=["dice"], nargs="*",
                        choices=["dice", "iou", "mse"])
    parser.add_argument("--batch_size", help="batch size for inference", default=8, type=positive_int)
    parser.add_argument("--threshold", help="threshold for binary metrics", type=restricted_float, default=0.5)

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print("Input directory not found!")
        exit()

    if not os.path.isdir(args.labels):
        print("Labels directory not found!")
        exit()

    data_generator = EvalStackGenerator(args.input, args.labels, batch_size=args.batch_size)

    if os.path.isdir(args.model):
        model_evals = {}
        for m_file in os.listdir(args.model):
            if m_file.endswith(".hdf5"):
                model = tf.keras.models.load_model(f"{args.model}/{m_file}", compile=False)
                model_evals[m_file] = evaluate_model(model, data_generator, metrics=args.metrics,
                                                     threshold=args.threshold).mean()
        print(model_evals)
    elif os.path.isfile(args.model):
        if not args.model.endswith(".hdf5"):
            print("Model file is not a hdf5 file!")
            exit()
        model = tf.keras.models.load_model(args.model)
        dices = evaluate_model(model, data_generator, metrics=args.metrics, threshold=args.threshold)
        print(dices.mean())
    else:
        print(f"model path / file not found!")
        exit()


def nope():
    return False


if __name__ == "__main__":
    # this ugly hack is here to prevent import optimization
    if nope() is True:
        scripts.resunet.res_unet(0, 0, 0, 0, 0)
    main()
