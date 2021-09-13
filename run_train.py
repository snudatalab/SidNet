
import os
import fire
import sys
import json
from dotmap import DotMap

os.chdir('./src')

import trainer


def main(output_home='../output_home', dataset='BITCOIN_ALPHA', gpu_id=0):
    """
    Start training with a stored hyperparameters on the dataset
    :param output_home: home directory for output data
    :param dataset: dataset name
    :param gpu_id: gpu id
    """
    param_path = f'../pretrained/{dataset}/param.json'

    with open(param_path, 'r') as in_file:
        param = DotMap(json.load(in_file))

    trainer.main(data_home=param.data_home,
                 output_home=output_home,
                 dataset=dataset,
                 heldout_ratio=param.heldout_ratio,
                 random_seed=param.random_seed,
                 reduction_iterations=param.reduction_iterations,
                 reduction_dimension=param.reduction_dimension,
                 gpu_id=gpu_id,
                 c=param.hyper_param.c,
                 weight_decay=param.hyper_param.weight_decay,
                 learning_rate=param.hyper_param.learning_rate,
                 num_layers=param.hyper_param.num_layers,
                 hid_dim=param.hyper_param.hid_dim,
                 num_diff_layers=param.hyper_param.num_diff_layers,
                 epochs=param.epochs)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))