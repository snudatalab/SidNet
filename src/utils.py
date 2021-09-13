from sklearn.metrics import roc_auc_score, f1_score
from dotmap import DotMap
from loguru import logger


def compute_accuracies(y, scores, predictions):
    """
    Compute accuracies in terms of AUC and F1 scores
    # 1 ==> positive, 0 ==> negative
    :param y: true labels
    :param scores: predicted scores
    :param predictions: predicted labels
    :return: auc and f1 scores
    """
    auc = roc_auc_score(y, scores)

    f1_scores = DotMap()
    f1_scores.macro  = f1_score(y, predictions, average='macro')
    f1_scores.micro  = f1_score(y, predictions, average='micro')
    f1_scores.binary = f1_score(y, predictions, average='binary')

    return auc, f1_scores


def log_param(param):
    """
    Print the input parameters into logger
    :param param: parameters
    """
    for key, value in param.items():
        if type(value) is DotMap:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))
