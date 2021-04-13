# Forschungspraktikum __*Embedded Hardware Systems Design*__ 2021
This repository contains the __CSLNet__ according to [Laina et al.](https://arxiv.org/abs/1703.10701) in a modified version 
so it can fit on the Ultra96-V2 accelerator board. Furthermore, it contains scripts for transferring the trained model to 
an appropriate accelerator board using the [Vitis AI workflow](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html).



## The CSL network
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
