# Komplexpraktikum __*Computer- und robotergestützte Chirurgie*__ 2020
This repository contains our 3 networks we worked with: 
- The __CSLNet__ according to [Laina et al.](https://arxiv.org/abs/1703.10701)
- A __MaskRCNN__  network using the [Detectron2](https://github.com/facebookresearch/detectron2) framework
- A __Combined__ network using the [Detectron2](https://github.com/facebookresearch/detectron2) framework which combines
the MaskRCNN network and the CSLNet by implementing a custom CSL head

## CSL
Run the CSL network with `python csl/run.py --argument2 --argument2  ...`. The `--command` argument has to be
 `init, train` or `call` and is always required.
#### init
Initializes the CSL model for further training. It basically loads the model with pretrained encoder weights and outputs
a __csl.pth__ file containing the model. With the `--workspace` argument you can specify the output directory for that file.
#### train
Starts training the CSL model. Possible arguments are:

`--workspace` Workspace directory containing the .pth model. <br>
`--dataset` Directory containing the annotated data (.json files) <br>
`--segloss` The segmentation loss. Either `ce` or `dice` <br>
`--normalize` Flag. If set, normalize gaussian kernels on ground truth heatmaps to \[0-1\] <br>
`--batch` The batch size <br>
`--lambdah` The lambda value according to the paper. Weighing parameter between segmentation and localisation <br>
`--sigma` The sigma value for the gaussian kernels applied to the ground truth heatmaps <br>
`--learningrate` The learning rate

#### call
Does inference on a trained model. Specify the dataset with `--dataset` and the directory containing the .pth file with
`--workspace`
## MaskRCNN
Before you can use the MaskRCNN network you have to register the dataset properly. The dataset directory has to contain
the images in .png format and their corresponding annotations in .json files. To register it, run: 
`python detectron2_commons/register_dataset.py --dataset [path/to/dataset/] --output [output/directory]`. This generates 
.json file describing the dataset.

Now you can run the network with `python maskrcnn/detectron_run.py --config [path/to/config] --dataset[path/to/dataset]
 [--train]`. The config file is a .yaml file according to the Detectron2 standard (like in maskrcnn/configs). The 
 dataset directory has to contain the previously generated .json file. The `--train` specifies whether to train or not.
 
 ## Combined
 Just as with the MaskRCNN network you have to register the dataset with the register_dataset.py script. Then start
 the network with `python combined/combined_run.py ...`. The arguments are identical to the MaskRCNN network.