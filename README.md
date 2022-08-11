# fMRI-S4

This the official Pytorch implementation of the paper [*fMRI-S4: Learning short- and long-range dynamic fMRI dependencies using 1D Convolutions and State Space Models*](https://arxiv.org/abs/2208.04166)

<p align="center">
  <img src=./images/model.png>
</p>

## Setup

### Requirements
This repository requires Python 3.8+ ,Pytorch 1.9+ and Pytorch Lightning
Other packages are listed in `requirements.txt`.

### Data

#### Datasets and Dataloaders
The Abide dataset can be requested from [here](http://preprocessed-connectomes-project.org/abide/)
The Mddrest dataset can be requested from [here](http://rfmri.org/REST-meta-MDD)

The Mddrest provide the pre-processed timecourses directly. For ABIDE you can generate the time-courses using nilearn.
You can see an example of how to parcellate the data and extract the timecourses  in 'data_notebook/parcellate.ipynb'
To integrate it directly to the code, you can organize the paths and into a csv file as in provided in the examples in 'csvfiles/'
All logic for creating and loading datasets to the model is in `datasets.py`.


### Cauchy Kernel

The S4 module relies on the "Cauchy kernel" described in the [paper](https://arxiv.org/abs/2111.00396).
The implementation of this requires a custom CUDA kernel.
To install it, Run `python setup.py install` from the directory `extensions/cauchy/`.


## Running Experiments

Examples of how to train the model using different configurations

```
python main.py  --dataset ABIDE --atlas HO --n_conv_layers 1 --n_s4_layers 2
python main.py --dataset Mddrest --atlas AAL --n_conv_layers 3 --n_s4_layers 0

```

The default parameters are described in the paper. However you can switch the and I observed robust performance using different hyper-parameters configurations. I recommended using the [weights&biases sweep](https://docs.wandb.ai/guides/sweeps) function for hyper-parameters search.

## Explainability

Not written in the paper. I have experimented with visualizing the salient ROIs using different methods in the [Captum](https://captum.ai/docs/introduction.html) Library.
Feature Permutation with a mask covering the entire temporal profile of regions seems to work best for me. However this needs more exploration.
I have tried with simple experiments where I create synthetic classes by classify fMRI with corrupted vs uncorrupted time-courses in certain regions and it seems to work fine.
See 'interpret.py' for an example.


## Citation
A large body of the code used for the S4 model was adopted from this [repo](https://github.com/HazyResearch/state-spaces), so please if you find any part of this work valuable cite their work along with ours.

```

@misc{https://doi.org/10.48550/arxiv.2208.04166,
  doi = {10.48550/ARXIV.2208.04166},
  
  url = {https://arxiv.org/abs/2208.04166},
  
  author = {El-Gazzar, Ahmed and Thomas, Rajat Mani and Van Wingen, Guido},
  
  keywords = {Machine Learning (cs.LG), Image and Video Processing (eess.IV), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {fMRI-S4: learning short- and long-range dynamic fMRI dependencies using 1D Convolutions and State Space Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}


@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R\'e, Christopher},
  booktitle={The International Conference on Learning Representations ({ICLR})},
  year={2022}
}

```
