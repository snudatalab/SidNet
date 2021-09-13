# SidNet

This is a PyTorch implementation of **Signed Diffusion Network**.

## Overview
How can we model node representations to accurately infer the signs of missing edges in a signed social graph?

Signed social graphs have attracted considerable attention to model trust relationships between people.
Various representation learning methods such as network embedding and graph convolutional network (GCN) have been proposed to analyze signed graphs.
However, existing network embedding models are not end-to-end for a specific task, and GCN-based models exhibit a performance degradation issue when their depth increases.

In this paper, we propose Signed Diffusion Network (SidNet), a novel graph neural network that achieves end-to-end node representation learning for link sign prediction in signed social graphs.
Our main contributions are summarized as follows:

* **Method.** We design SidNet, an end-to-end  representation learning method in a signed graph with multiple signed diffusion layers. Our signed diffusion layer exploits signed random walks to propagate node embeddings on signed edges, and injects local features. This enables SidNet to learn distinguishable node embeddings effectively considering multi-hop neighbors while preserving local information.
* **Analysis.** We theoretically analyze the convergence property of our signed diffusion layer, showing how SidNet prevents the over-smoothing issue.
We also provide the time complexity analysis of SidNet, showing SidNet is linearly scalable w.r.t. the numbers of edges.
* **Experiments.** Extensive experiments show that SidNet effectively learns node representations of signed social graphs for link sign prediction, giving at least 3.3% higher accuracy than the state-of-the-art models in real datasets.




## Prerequisites
* python 3.6+
* torch 1.5.0
* numpy 1.18.1
* scipy 1.4.1
* scikit_learn 0.23.1
* tqdm 4.46.1
* fire 0.3.1
* pytictoc 1.5.0
* dotmap 1.3.17
* loguru 0.5.0


## Datasets and Pre-trained SidNet
We provide the datasets used in the paper for reproducibility.
You can find the datasets at `./data/${DATASET}` folder where the file's name is `data.tsv`. 
* ${DATASET} is one of `BITCOIN_ALPHA`, `BITCOIN_OTC`, `SLASHDOT` and `EPINIONS`.

The file contains the list of signed edges where each line represents a tuple of (src, dst, sign) which is tab-separated. 
There are four real-world signed social networks:
* `BITCOIN_ALPHA`: signed social network from the Bitcoin Alpha platform [[link]](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
* `BITCOIN_OTC`: signed social network from the Bitcoin OTC platform [[link]](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
* `WIKIPEDIA`: signed network representing the administrator election procedure in Wikipedia [[link]](http://konect.cc/networks/elec/)
* `SLASHDOT`: signed social network from the Slashdot online review site [[link]](http://konect.cc/networks/slashdot-zoo/)
* `EPINIONS`: signed social network from the Epinions online review site [[link]](http://konect.cc/networks/epinions/)

This repository also contains pre-trained SidNet models; you can find them in `./pretrained/${DATASET}` folder where the file's name is `model.pt`. 
The hyperparameters used for training an SidNet model are saved at `param.json`.   

## Simple Demo
You can run the demo script by `bash demo.sh`. 
It trains SidNet on `BITCOIN_ALPHA` dataset with hyperparameters stored at `./pretrained/BITCOIN_ALPHA/param.json`.
This demo saves the trained model at `./output/BITCOIN_ALPHA/model.pt`.
Then, it evaluates the trained model in terms of AUC and F1-macro scores. 

## Results of Pre-trained SidNet
We provide pre-trained SidNet models which are stored at `./pretrained/${DATASET}/model.pt`, respectively.
The experimental results with the pre-trained models are as follows:

| **Dataset**       | **AUC** | **F1-macro** |
| ----------------- | ------- | ------------ |
| **Bitcoin-Alpha** | 0.9139  | 0.7428       |
| **Bitcoin-OTC**   | 0.9227  | 0.8060       |
| **Wikipedia**     | 0.9094  | 0.8011       |
| **Slashdot**      | 0.8944  | 0.7792       |
| **Epinions**      | 0.9397  | 0.8521       |

Note that we conducted the experiments on GTX 1080 Ti (CUDA version 10.1), and the above results were produced with `random-seed=1`. 


### Used Hyperparameters 
We briefly summarize the hyperparameters used in the above results. 
The hyperparameters are stored at `./pretrained/${DATASET}/param.json`.

* Hyperparameters of SidNet
    - `num-layers` (L): number of layers
    - `c`: ratio of local feature injection
    - `num-diff-layers` (K): number of diffusion steps
    - `hid-dim` (d): hidden feature dimension

| **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `num-layers` (L) | 2 | 2 | 2 | 2 | 2 |
| `c` | 0.35 | 0.25 | 0.45 | 0.55 | 0.55 |
| `num-diff-layers` (K) | 10 | 10 | 10 | 10 | 10 |
| `hid-dim` (d) | 32 | 32 | 32 | 32 | 32 |

* Hyperparameters of optimizer
    - optimizer: Adam
    - L2 regularizer (`weight-decay`, λ): 1e-3
    - `learning-rate`: 0.01
    - `epochs`: 100

* Input feature dimension (`reduction-dimension`): 128

### How to Reproduce the Above Results with the Pre-trained Models
You can reproduce the results with the following command which evaluates a test dataset using a pre-trained model. 

```shell
python3 -m run_eval --input-home ../pretrained --dataset ${DATASET} --gpu-id ${GPU_ID}
```
* ${DATASET} is one of `BITCOIN_ALPHA`, `BITCOIN_OTC`, `SLASHDOT` and `EPINIONS`.
* ${GPU_ID} is `-1` or your gpu id (maybe, non-negative integer) where `-1` indicates it runs on CPU.

The pre-trained models were generated by the following command:
```shell
python3 -m run_train --output-home ../output --dataset ${DATASET} --gpu-id ${GPU_ID}
```

Note that those scripts automatically find the file `param.json` for the above hyperparameters.
To tune the hyperparameters, modify the json file, or use the below commands.

## Detailed Usage
You can train and evaluate your own datasets using `trainer.py` and `evaluator.py`, respectively.
To use those scripts properly, move your working directory to `./src`.

### Training
The following command performs the training process of SidNet on a given dataset. 
This automatically splits the dataset by `heldout-ratio`, e.g., if it is 0.2, training:test=0.8:0.2.
Note that the split data are guaranteed to be the same if the same `random-seed` is given.
After the training is completed, it generates two files called `model.pt` and `param.json` at the `${output-home}/${dataset}` folder where `model.pt` conatins parameters of the trained model, and `param.json` has hyperparameters used for the model.

```shell
python3 -m trainer \
    --data-home ../data \
    --output-home ../output \
    --dataset BITCOIN_ALPHA \
    --heldout-ratio 0.2 \
    --radnom-seed 1 \
    --reduction-dimension 128 \
    --reduction-iterations 30 \
    --gpu-id 0 \
    --c 0.15 \
    --weight-decay 1e-3 \
    --learning-rate 0.01 \
    --num-layers 1 \
    --hid-dim 32 \
    --num-diff-layers 10 \
    --epochs 100
```

| **Option** | **Description** | **Default** |
|:--- | :--- | :---: |
|`data-home`| data directory path | ../data |
|`output-home`| output directory path | ../output |
|`dataset`| dataset name | BITCOIN_ALPHA |
|`heldout-ratio`| heldout ratio between training and test | 0.2 |
|`radnom-seed`| random seed used for dataset split and torch | 1 |
|`use-torch-random-seed`| whether torch uses the above random seed | True |
|`reduction-dimension`| input feature dimension (SVD) | 128 |
|`reduction-iterations`| number of iterations required by SVD computation  | 30 |
|`gpu-id`| gpu id | 0   |
|`c`| ratio of local feature injection | 0.15  |
|`weight-decay`|weight decay (L2 regularizer) for optimizer | 1e-3 |
|`learning-rate`| learning rate for optimizer| 0.01|
|`num-layers`| number of layers (L)| 1 |
|`hid-dim`| hidden feature dimension (d)| 32 |
|`num-diff-layers`| number of diffusion steps (K)| 10 |
|`epochs`| target number of epochs | 100 |


### Evaluation
This performs the evaluation process of SidNet, and reports AUC and F1-macro scores on the test dataset.
This uses `model.pt` and `param.json`; thus, you need to check if they are properly generated by `trainer.py` before this evaluation.
Note that it uses the same random seed used by `trainer.py` so that the test dataset is valid for this evaluation.

```shell
python3 -m evaluator \
    --input-home ../output \
    --dataset BITCOIN_ALPHA \
    --gpu-id 0
```

| **Option** | **Description** | **Default** |
|:--- | :--- | :---: |
|`input-home`|  directory where a trained model is stored | ../output |
|`dataset`| dataset name | BITCOIN_ALPHA |
|`gpu-id`| gpu id | 0   |

