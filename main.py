#!/usr/bin/env python
import argparse
import json
import os.path
import shutil
from json import JSONDecodeError
from os.path import exists

import cv2
import numpy as np
import pyvista as pv
import tensorflow as tf
from matplotlib.cm import get_cmap
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage import io
from tensorflow import keras

from scripts.postprocess import create_mask, ema_stack, apply_mask, apply_mask_stack, force_convex
from scripts.renderutil import mesh_from_volume
from scripts.util import progress_bar, log_msg, save_imagej_composite, quantile_normalize, restricted_float, \
    positive_float, positive_int, fast_label_volume
from scripts.watershed import distance_transform_watershed


# quick helper function for code readability
def should_post_process(config, option, channel):
    return option in config["models"][channel]["post_process"]


def main():
    # Hide TF logging.
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Segment tomograms using models listen in configuration file. "
                                                 "Segmentations can be rendered using PyVista, "
                                                 "and metrics such as volume can be displayed for each channel. ")
    parser.add_argument("input",
                        help="input file for segmentation or rendering. It may also be a scene directory (mesh_cache) "
                             "if using 'render' mode.")
    parser.add_argument("-o", "--out", help="output directory", default="out")
    parser.add_argument("-m", "--mode",
                        help="Program mode. 'segment' takes a tomogram input and segments it. To include render add "
                             "the --render flag. 'render' takes a segmentation and renders it.", default='segment',
                        choices=['segment', 'render'])
    parser.add_argument("--config", help="config file to use.", default='default-config.json')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--raw", help="save an additional copy without post-processing.",
                        action="store_true")
    parser.add_argument("--norender", help="Disable rendering and only save segmentation",
                        action="store_true")
    parser.add_argument("--hiderender", help="Prevents rendering from displaying (only saved). "
                                             "To disable rendering generation, use --norender",
                        action="store_true")
    parser.add_argument("-u", "--unit", help="volume unit for segmentation measurement", default='µm³'),
    parser.add_argument("-s", "--size", help="volume of a single voxel in given units (see --unit)", default=7.536e-6,
                        type=positive_float),
    parser.add_argument("-d", "--dim", "--dimensions",
                        help="original dimensions of a segmentation. "
                             "Used to rescale renderings back to original proportions.",
                        nargs=2, default=[512, 512], type=positive_int)
    parser.add_argument("--includes_gray", help="Set to true if rendering a segmentation with a grey channel at the end"
                                                " which should not me rendered", action="store_true")

    args = parser.parse_args()
    args.scale = (args.dim[0] / 512, args.dim[1] / 512)

    if not exists(args.config):
        # raise FileNotFoundError(f"Config file '{args.config}' not found")
        exit(f"Config file '{args.config}' not found")
    cfg_file = open(args.config)  # if this fails, just let it raise the error.

    config = None  # fix IDE complaints.
    try:
        config = json.load(cfg_file)
    except JSONDecodeError:
        # raise ValueError("Config corrupt, failed to decode JSON.")
        exit("Config corrupt, failed to decode JSON.")

    # make dirs
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "segment":
        if not exists(args.input):
            # raise FileNotFoundError(f"Input file '{args.input}' not found")
            exit(f"Input file '{args.input}' not found")
        if not args.input.endswith(".tif") and not args.input.endswith(".tiff"):
            # raise ValueError("Input file is not a TIFF file")
            exit("Input file is not a TIFF file")

        tomo_volume = io.imread(args.input).astype('float32')
        if len(tomo_volume.shape) != 3:
            # raise ValueError("Input file must be 3D")
            exit("Input file must be 3D")
        semantice_seg, instance_seg = segment_volume(tomo_volume, args, config)
        if not args.norender:
            render_segmentation(instance_seg, args, config)

    elif args.mode == 'render':
        filename = os.path.splitext(os.path.basename(args.input))[0]
        mesh_dir = f"{args.out}/.mesh_cache_{filename}"
        if exists(mesh_dir):
            render_scene(mesh_dir)
        else:
            if not exists(args.input):
                # raise FileNotFoundError(f"Input file '{args.input}' not found")
                exit(f"Input file '{args.input}' not found")
            if not args.input.endswith(".tif") and not args.input.endswith(".tiff"):
                # raise ValueError("Input file is not a TIFF file")
                exit("Input file is not a TIFF file")
            segmentation = io.imread(args.input).astype('float32')
            segmentation = np.rollaxis(segmentation, 3, 1)  # loaded as ZXYC
            print(segmentation.shape)
            render_segmentation(segmentation, args, config)


def segment_volume(tomo_volume, args, config):
    """
    Takes a reconstructed tomogram z-stack and returns a stack with segmentations based on config
    Parameters
    ----------
    tomo_volume : np.array
        The tomogram z-stack volume
    args : Namespace
        Arguments from the argument parser
    config : dict
        Dictionary containing configuration
    Returns
    -------
    np.array
        A composite of the original array and segmentations
    """

    filename = os.path.basename(args.input)
    original_shape = tomo_volume.shape
    stack_size = original_shape[0]
    assert (stack_size >= 5)

    # in case the user didn't specify, and we have to do resizing, update the params here.
    if tomo_volume.shape[1:3] != (512, 512):
        args.scale = (tomo_volume.shape[1] / 512, tomo_volume.shape[2] / 512)

    preprocessed_volume = np.zeros((512, 512, stack_size))
    num_channels = len(config['models'])
    output_volume = np.zeros((stack_size - 4, num_channels + 1, 512, 512)).astype('float32')  # ZCXY
    instance_vol = np.zeros((stack_size - 4, num_channels, 512, 512)).astype(np.uint8)

    # rescale and normalize input
    log_msg("Preprocessing...")
    progress_bar(0, stack_size)
    for i in range(stack_size):
        preprocessed_volume[:, :, i] = cv2.resize(tomo_volume[i], (512, 512), interpolation=cv2.INTER_CUBIC)
        preprocessed_volume[:, :, i] = quantile_normalize(preprocessed_volume[:, :, i])
        progress_bar(i + 1, stack_size)

    # predict for each channel
    for chan, model_info in enumerate(config['models']):
        log_msg(f"Processing {model_info['name']}")
        model = keras.models.load_model(f"models/{model_info['hdf5']}", compile=False)
        progress_bar(0, stack_size - 4)
        for i in range(stack_size - 4):
            pred_input = np.copy(preprocessed_volume[:, :, i:i + 5]).reshape((1, 512, 512, 5))
            output_volume[i, chan] = model.predict(pred_input, verbose=0).reshape((512, 512))
            progress_bar(i + 1, stack_size - 4)

    # add original image (without edge frames) to output stack.
    # roll axis, since tensorflow and tifffile/imagej disagree
    output_volume[:, num_channels, :, :] = np.rollaxis(preprocessed_volume[:, :, 2:-2], 2)
    output_volume = output_volume.astype('float32')
    binary_volume = np.zeros(output_volume.shape).astype(np.uint8)

    colors = []
    for model_info in config['models']:
        colors.append(model_info["imagej_col"])
    colors.append("gray")  # last channel is gray

    if args.raw:  # this feature is only available for default case.
        save_imagej_composite(f"{args.out}/raw_{filename}", output_volume, colors)

    log_msg(f"Post-processing and saving segmentations...")
    mask = create_mask(output_volume[:, 0, :, :], alpha=0.05)  # first channel for outlier detection
    for chan, model_info in enumerate(config['models']):
        if args.raw:  # if user wants raw output, it's now or never
            io.imsave(f"{args.out}/raw_{chan}_{filename}", output_volume[:, chan, :, :])

        if should_post_process(config, "o", chan):
            # apply mask generated earlier from channel 0 for outlier removal
            output_volume[:, chan, :, :] = apply_mask(output_volume[:, chan, :, :], mask)

        if should_post_process(config, "a", chan):
            output_volume[:, chan, :, :] = gaussian_filter(output_volume[:, chan, :, :], sigma=3)

        # thresholding
        binary_volume[:, chan, :, :] = ((output_volume[:, chan, :, :] > model_info['threshold']) * np.ones(
            (stack_size - 4, 512, 512))).astype(np.uint8)

        if should_post_process(config, "h", chan):
            # do 3d instance seg to help with instance seg
            binary_volume[:, chan, :, :] = binary_fill_holes(binary_volume[:, chan, :, :])

        log_msg("Conducting instance segmentation")

        if should_post_process(config, "f", chan):  # connectivity based
            labeled_vol, num_labels = fast_label_volume(binary_volume[:, chan, :, :])
            labeled_vol = labeled_vol.astype(np.uint8)
        else:
            dynamic = 2
            if model_info['min_voxels'] >= 500:
                dynamic = 8
            labeled_vol = distance_transform_watershed(binary_volume[:, chan, :, :], dynamic=dynamic,
                                                       full_connectivity=True).astype(int)
            num_labels = np.max(labeled_vol)
        # log_msg("merging strongly connected volumes")
        # labeled_vol, num_labels = merge_connected_volumes(labeled_vol, area_ratio=0.8)
        log_msg(f"Finished instance segmentation for {model_info['name'].lower()}")
        for i in range(1, num_labels + 1):
            filtered_vol = (labeled_vol == i).astype(np.uint8)
            # check if we have volume minimum and enough voxels to continue if so.
            num_voxels = np.sum(filtered_vol)
            if should_post_process(config, "v", chan) and num_voxels < model_info['min_voxels']:
                pass
            else:  # if we don't have volume minimum or we meet the threshold, continue
                if should_post_process(config, "c", chan):
                    filtered_vol = force_convex(filtered_vol)
                # fill in Z direction, since it is most susceptible to missing cone.
                if should_post_process(config, "h", chan):
                    for j in range(len(binary_volume)):
                        binary_volume[j, chan, :, :] = binary_fill_holes(binary_volume[j, chan, :, :])

                instance_vol[:, chan, :, :] = (instance_vol[:, chan, :, :] + (filtered_vol * i))

        binary_volume[:, chan, :, :] = (instance_vol[:, chan, :, :] > 0)
        if should_post_process(config, "m", chan):
            # apply channel 0 using specified thresholding (without applying it first)
            binary_volume[:, chan, :, :] = apply_mask_stack(binary_volume[:, chan, :, :], binary_volume[:, 0, :, :])
            instance_vol[:, chan, :, :] = apply_mask_stack(instance_vol[:, chan, :, :], binary_volume[:, 0, :, :])

        log_msg(f"{model_info['name']} volume: "
                f"{round(np.sum(binary_volume[:, chan, :, :]) * args.size, 3)}{args.unit}")
        # save output
        io.imsave(f"{args.out}/{chan}_{filename}", binary_volume[:, chan, :, :])

    save_imagej_composite(f"{args.out}/semantic_composite_{filename}", binary_volume, colors)
    colors.remove("gray")
    save_imagej_composite(f"{args.out}/instance_composite_{filename}", instance_vol, colors)
    log_msg(f"Segmentation complete.")
    return output_volume, instance_vol


def render_segmentation(seg_volume, args, config):
    """
    Takes a z-stack with multiple channels where the last channel is the original and
    the preceding are segmentation channels, and renders it using PyVista
    ----------
    seg_volume : np.array
        The segmentation and tomogram z-stack volume
    args : Namespace
        Arguments from the argument parser
    config : dict
        Dictionary containing configuration
    """
    p = pv.Plotter(lighting='three lights')
    p.set_background('black', top='white')

    # save files
    manifest = []
    filename = os.path.basename(args.input)[:-4]
    mesh_dir = f"{args.out}/.mesh_cache_{filename}"
    os.makedirs(mesh_dir, exist_ok=True)
    num_channels = seg_volume.shape[1]
    if args.includes_gray:
        num_channels -= 1

    log_msg("Generating meshes...")
    for chan in range(num_channels):
        labeled_vol = seg_volume[:, chan, :, :]
        num_labels = int(np.max(labeled_vol))
        chan_measured_vol = 0
        model_cfg = config['models'][chan]
        log_msg(f"Generating for {model_cfg['name'].lower()}...")
        progress_bar(0, num_labels)
        for i in range(1, num_labels + 1):
            filtered_vol = (labeled_vol == i).astype(np.uint8)
            num_voxels = np.sum(filtered_vol)
            if num_voxels > 0:
                col = get_cmap('hsv')(i / num_labels)
                if chan == 0 and False:
                    col = "white"

                col = model_cfg["pv_col"]
                mesh = mesh_from_volume(filtered_vol, force_sphere=model_cfg["convert_sphere"], scale=args.scale)
                # flip such that image coords match the 3D coords
                mesh = mesh.flip_y(point=[0.0, 0.0, 0.0], inplace=False)
                if should_post_process(config, "s", chan):
                    mesh = mesh.smooth(n_iter=500)
                chan_measured_vol += mesh.volume
                p.add_mesh(mesh, color=col, opacity=model_cfg["opacity"], diffuse=0.5, specular=0.5, ambient=0.5)
                # save the mesh and add to manifest
                mesh_filename = f"{model_cfg['name'].lower().replace(' ', '_')}_{i}.vtk"
                mesh.save(f"{mesh_dir}/{mesh_filename}")
                manifest.append(
                    {"file": mesh_filename, "color": col, "opacity": model_cfg["opacity"], "volume": mesh.volume})
            progress_bar(i, num_labels)  # progress either way.
        log_msg(
            f"{model_cfg['name']} generated "
            f"with volume of {round(chan_measured_vol * args.size, 3)}{args.unit}")

    with open(f"{mesh_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f)
        shutil.copyfile("scripts/mesh-render-script.py", f"{mesh_dir}/render-segmentation.py")
    log_msg("Saving scene...")

    if not args.hiderender:
        log_msg("Opening PyVista rendering...")
        p.show_bounds(grid='front', location='outer', all_edges=True, )
        p.show_axes()
        p.show()


def render_scene(fol):
    """
    Loads VTK mesh objects from a directory and adds them according to a manifest.
    ----------
    dir : str
        Directory containing manifest and .vtk files
    """
    f_manifest = open(f"{fol}/manifest.json")
    manifest = json.load(f_manifest)
    p = pv.Plotter(lighting='three lights')
    p.set_background('black', top='white')
    log_msg("Loading saved meshes...")
    for mesh_data in manifest:
        mesh = pv.read(f"{fol}/{mesh_data['file']}")
        p.add_mesh(mesh, color=mesh_data['color'], opacity=mesh_data['opacity'], diffuse=0.5, specular=0.5,
                   ambient=0.5)

    log_msg("Opening PyVista rendering...")
    p.show_bounds(grid='front', location='outer', all_edges=True, )
    p.show_axes()
    p.show()


if __name__ == "__main__":
    main()
