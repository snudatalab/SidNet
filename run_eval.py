
import os
import fire
import sys
import json
from dotmap import DotMap
os.chdir('./src')

import evaluator


def main(input_home='../pretrained',
         dataset='BITCOIN_ALPHA',
         gpu_id=0):
    """
    Start evaluating the stored model with a stored hyperparameters on the dataset
    :param input_home: home directory for input data (model, hyperparameters)
    :param dataset: dataset name
    :param gpu_id: gpu id
    """

    evaluator.main(input_home=input_home,
                   dataset=dataset,
                   gpu_id=gpu_id)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
