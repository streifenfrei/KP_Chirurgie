# Important for the lab runnings

Folder for the project is located in 
`mnt/g27prist/TCO/TCO-Studenten/chernykhalex/`

To run maskrcnn: `python3 detectron_run.py`

1. Check the cuda version installed on the pc 
`nvidia-smi`
2. find a **right wheel** for the detectron according to the required Cuda-version

For **Cuda 10.1** = `https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html`

For **Cuda 10.2** = `https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html`


Torch version
```bash
Torch == 1.5
torchvision ==0.6.0
```