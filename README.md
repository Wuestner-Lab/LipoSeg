# Yeast cryo-SXT instance segmentation pipeline

## Introduction

This software generates instance segmentations from yeast cell tomograms. The pipeline / workflow can be converted to
work with other cells and conditions. There are models for cell membrane, vacuole and lipid droplet segmentation. This
makes it ideal for quantifying lipophagy in *Saccharomyces cerevisiae*.

For full details see our [paper]:
'*Automated quantification of lipophagy in Saccharomyces cerevisiae from fluorescence and cryo-soft X-ray microscopy
data using deep learning*' by Egebjerg et al.

![This will show figure text][pipeline-figure]

[pipeline-figure]: https://i.imgur.com/MrwwpZZ.png

[paper]: https://www.doi.org/

## Installation

Pull or download the code to the directory of your choice.

You can create a new environment for package installation. e.g. with conda:

```bash
conda create --name <env_name> --file requirements.txt python=3.8 ipython
conda activate <env_name>
```

Or install the required packages using pip for your default Python.

```bash
pip install requirements.txt
```

For NVidia card users, see TensorFlow for details on CUDA setup https://www.tensorflow.org/install/pip.

## Usage

The program is run as a commandline Python script.

Only TIFF files are currently accepted. For other formats, software like ImageJ can be used to convert to .tif

The below examples demonstrate various usages.

Segment and render an input stack.

```bash
python main.py input.tif -o output_dir
```

Render a segmentation with original dimensions of 650x600 and pixel width of 0.02µm

```bash
python main.py segmentation.tif -o output_dir --mode render --dim 650 600 --size 0.02
```

Segment and generate meshes without rendering.

```bash
python main.py input.tif -o output_dir --hiderender
```

You can also execute the script directly as such

```bash
./main.py input.tif -o output_dir
```

## Parameters and configuration

### Parameters

This can also be found using the --help flag.

```bash
python main.py --help
```

| Option | Description | default |
| --- | --- | --- |
| help | Displays all arguments and their meaning |
| out | The output directory path | ./out |
| mode | Set program mode to 'render' or 'segment'. Render mode will take segmentations as input and segment mode will take raw stacks. | segment |
| includes_gray | Specifies that a segmentation used in 'render' mode, includes gray channel, which should not be rendered.
| config | Path of the configuration file | default-config.json |
| verbose | Increase logging verbosity |  |
| norender | Only perform segmentation and do not generate mesh |  |
| hiderender | Generate and save mesh, but do not render |  |
| raw | Save a copy of the raw model predictions | |
| unit | Pixel width / height unit | µm |
| size | Pixel width / height in given unit | 0.0196 |
| dim | XY dimensions of original image stack | 512, 512 |

### Configuration

Each channel is configured individually. It is recommended that you copy the default-configuration rather than change it
directly. The configuration specifies which models to include and additional parameters for post-processing and visual
aspects.

| Option | Description  |
| --- | --- |
| name | Name of the channel |
| hdf5 | File name of the model (relative to ./models) |
| imagej_col | Color of channel in semantic segmentation composite tiff |
| pv_col | Color in PyVista rendering |
| opacity | Opacity in PyVista rendering |
| convert_sphere | Converts instances to spheres based on largest slice |
| min_voxels | Minimum voxels for volume threshold option |
| threshold | Threshold for CNN prediction |
| post_process | Post-processing flags (see below) |

These are the flags for post-processing. See for [paper] for more details.

| Flag | Description |
| --- | --- |
| o | Outlier removal |
| m | Channel masking |
| a | Averaging / voxel blurring |
| h | Hole filling |
| c | Convex constraint |
| v | Volume thresholding |
| s | Surface smoothing |
| f | Fast connectivity-based instance segmentation |

## Training new models

New models can be trained from the bottom up or by using another model as a pretrained U-Net. Trainer.py has been
included for this task. For mor details on the trainer.py script, run it with the --help flag. For more details on
implementation see [paper].

For custom architectures, ensure 512x512x5 input and 512x512x1 output for each model.

Training data is structured into biological samples with identical subfolders and filenames for inputs and labels,
respectively.

```
📁inputs
├── 📁sample_1
│   ├── 🔬image_a.tif
│   └── 🔬image_b.tif
└── 📁sample_2
    └── 🔬image_a.tif
    
📁labels
├── 📁sample_1
│   ├── 🏷image_a.tif
│   └── 🏷image_b.tif
└── 📁sample_2
    └── 🏷image_a.tif
```

## Questions / contributions

Contributions are welcome (especially training data). Questions can be sent to jegebjerg(at)bmb.sdu.dk

## Authors and acknowledgement

Jacob Marcus Egebjerg¹², Maria Szomek¹, Katja Thaysen¹, Alice Dupont Juhl¹, Stephan Werner³, Gerd Schneider³, Christoph
Pratsch³, Richard Röttger², Daniel Wüstner¹

¹Department of Biochemistry and Molecular Biology and ²Department of Mathematics and Computer Science, University of
Southern Denmark

³Department of X‑Ray Microscopy, Helmholtz-Zentrum Berlin  
\
\
If you found our work useful, please consider citing our [paper].

## License

[MIT](https://choosealicense.com/licenses/mit/)