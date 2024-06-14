## Image retrieval task using RaMBO
### Dataset
CUB-200-2011 dataset is available at [here](https://www.vision.caltech.edu/datasets/cub_200_2011/)
SOP dataset is available at [here](https://cvgl.stanford.edu/projects/lifted_struct/)

The datasets should be unzipped and placed to the `dataset/` folder.

### Training
The training configs are defined in `scripts/RaMBO.yaml`. Default configs are:
```
dataset: CUB200
batch_size: 64
lambda: 0.2
learning_rate: 0.000005
weight_decay: 0.0004
epoch: 80
```
Use `python train_host.py` to train on CUB-200-2011 dataset, if you want to train on SOP dataset, change the dataset inside the yaml file to `SOP`. The best model checkpoint and checkpoint of the current epoch will be save to `checkpoint/` folder.

### Testing
Use `python eval_host.py` to get the recall results on CUB-200-2011 dataset, if you want to experiment with SOP dataset, change line 42 of the file to:
```
dataset = "SOP"          # change dataset here for different evaluation settings
```