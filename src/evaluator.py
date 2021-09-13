from method.sidnet.train import SidNetTrainer
from dotmap import DotMap
import torch
import json
from data.data_loader import DataLoader
from method.sidnet.train import SidNetTrainer
from method.sidnet.model import SidNet
from loguru import logger
import sys
import fire
import utils


def main(input_home='../output',
         dataset='BITCOIN_ALPHA',
         gpu_id=0):
    """
    Evaluate SidNet
    :param input_home: directory where a trained model is stored
    :param dataset: dataset name
    :param gpu_id: gpu id
    """

    device = torch.device(f"cuda:{gpu_id}"
                          if (torch.cuda.is_available() and gpu_id >= 0)
                          else "cpu")

    param_output_path = f'{input_home}/{dataset}/param.json'
    model_output_path = f'{input_home}/{dataset}/model.pt'

    with open(param_output_path, 'r') as in_file:
        param = DotMap(json.load(in_file))
        param.device = device

        if param.use_torch_random_seed:
            torch.manual_seed(param.torch_seed)

        #
        data_loader = DataLoader(random_seed=param.random_seed,
                                 reduction_dimension=param.reduction_dimension,
                                 reduction_iterations=param.reduction_iterations)

        # data = {train, test}, train = {X, y}, test = {X, y} according to heldout_ratio
        data = data_loader.load(data_path=param.data_path,
                                heldout_ratio=param.heldout_ratio)

        trainer = SidNetTrainer(param)
        hyper_param = param.hyper_param
        converted_data = trainer.convert_data(data)

        model = SidNet(hid_dims=hyper_param.hid_dims,
                       in_dim=hyper_param.in_dim,
                       device=device,
                       num_nodes=converted_data.num_nodes,
                       num_layers=hyper_param.num_layers,
                       num_diff_layers=hyper_param.num_diff_layers,
                       c=hyper_param.c).to(device)

        model.load_state_dict(torch.load(model_output_path, map_location=device))

        loss = model(nApT=converted_data.train.nApT,
                     nAmT=converted_data.train.nAmT,
                     X=converted_data.H,
                     edges=converted_data.train.edges,
                     y=converted_data.train.y)

        model.eval()
        auc, f1_scores, _ = model.evaluate(test_edges=converted_data.test.edges,
                                           test_y=converted_data.test.y)

        logger.info('test auc: {:.4f}'.format(auc))
        logger.info('test f1_macro:  {:.4f}'.format(f1_scores.macro))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
