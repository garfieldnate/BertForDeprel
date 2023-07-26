import pandas as pd
from torch import gather, Tensor
from torch.nn import CrossEntropyLoss
from torch import sum as tsum

from .load_data_utils import DUMMY_ID

def deprel_pred_for_heads(deprels_pred: Tensor, heads_pred: Tensor):
    """
    Given the dependency relation label score predictions for all possible heads of each word and
    the list of predicted heads, return the scores with the head dimension removed, and just the
    scores for the labels on the predicted dependency-head arcs remaining.

    deprels_pred: tensor with 4 dimensions: (batch_len, n_class_deprel, seq_len, seq_len).
    Read indexing as [sentence_index][deprel_label_index][dependent_index][head_index], with the
    last dimension containing the scores for each potential head of the dependent.

    heads_pred: tensor with two dimensions: (batch_len, seq_len). Read indexing as
    [sentence_index][dependent_index], with the value at each index being the predicted head
    index.

    returns: tensor of size (batch_len, n_class_deprel, seq_len). Read indexing as
    [sentence_index][deprel_label_index][dependent_index], with the value at each index being
    the score for the given dependency relation label for the dependency edge between a
    dependent and its predicted head (which was specified in heads_pred, and is not contained
    explicitly in the output tensor).

    See test case if it's still not clear what this function does.
    """
    # modify heads_true to have the same shape as deprels_pred
    # add two dimensions of size 1: (batch_size, 1, 1, seq_len)
    heads_pred = heads_pred.unsqueeze(1).unsqueeze(2)
    # expand to (batch_size, n_class_deprel, 1, seq_len)
    heads_pred = heads_pred.expand(-1, deprels_pred.size(1), -1, -1).clone()
    heads_pred[heads_pred == DUMMY_ID] = 0
    # deprels_pred.shape after gather =  torch.Size([batch_len, n_class_deprel, 1, sq_len])
    # deprels_pred.shape after squeeze =  torch.Size([batch_len, n_class_deprel, seq_len])
    deprels_pred = gather(deprels_pred, 2, heads_pred).squeeze(2)

    return deprels_pred

def compute_loss_head(heads_pred: Tensor, heads_true: Tensor, criterion: CrossEntropyLoss):
    return criterion.forward(heads_pred, heads_true)

def compute_loss_deprel(deprels_pred: Tensor, deprels_true: Tensor, heads_true, criterion: CrossEntropyLoss):
    deprels_pred = deprel_pred_for_heads(deprels_pred, heads_true)
    return criterion.forward(deprels_pred, deprels_true)

def compute_loss_class(class_pred: Tensor, class_true: Tensor, criterion: CrossEntropyLoss):
    """This applies to any of the 1-dimensional classification tasks"""
    # todo: why do we permute here?
    return criterion.forward(class_pred.permute(0,2,1), class_true)

def __sum_2d_tensor(tensor: Tensor):
    return float(tsum(tensor, dim=(0,1)).item())


def compute_acc_head(heads_pred: Tensor, heads_true: Tensor, eps=1e-10):
    """
    heads_pred: torch.Size([batch_len, seq_len, seq_len])
    heads_true: torch.Size([batch_len, seq_len])
    """
    mask = (heads_true != DUMMY_ID)
    total = __sum_2d_tensor(mask) + eps
    correct = float(sum(heads_true[mask] == heads_pred.max(dim=1)[1][mask]))
    return correct, total

def compute_acc_class(class_pred: Tensor, class_true: Tensor, eps=1e-10):
    """This applies to any of the 1-dimensional classification tasks
    class_pred: torch.Size([batch_len, seq_len, n_class])
    class_true: torch.Size([batch_len, seq_len])
    """
    mask = (class_true != DUMMY_ID)
    total = __sum_2d_tensor(mask) + eps
    correct = float(sum(class_true[mask] == class_pred.max(dim=2)[1][mask]))
    return correct, total

def compute_acc_deprel(deprels_pred, deprels_true, heads_true, eps=1e-10):
    mask = (heads_true != DUMMY_ID)
    total = __sum_2d_tensor(mask) + eps
    deprels_pred = deprel_pred_for_heads(deprels_pred, heads_true)

    correct = float(sum(deprels_pred.max(dim=1)[1][mask] == deprels_true[mask]))
    # TODO :
    # - find better formula for summing up
    return correct, total

def compute_LAS(heads_pred: Tensor, deprels_pred: Tensor, heads_true: Tensor, deprels_true: Tensor):
    """Labled Attachment Score measures the accuracy of labeled dependency edges.
    Returns (number correct, total) indicating the number of correctly labeled dependency edges
    and the total number of edges predicted.
    heads_pred: torch.Size([batch_len, seq_len, seq_len])
    deprels_pred: torch.Size([batch_len, n_class_deprel, seq_len, seq_len])
    heads_true: torch.Size([batch_len, seq_len])
    deprels_true: torch.Size([batch_len, seq_len])
    """
    deprels_pred = deprel_pred_for_heads(deprels_pred, heads_true)

    # Ignore predictions for the tokens that don't begin a real word (CLS, subwords, padding, etc.)
    mask = (heads_true != DUMMY_ID)
    total = __sum_2d_tensor(mask)
    correct_head = heads_pred.max(dim=1)[1][mask] == heads_true[mask]
    correct_deprel = deprels_pred.max(dim=1)[1][mask] == deprels_true[mask]

    correct_LAS = tsum(correct_head & correct_deprel).item()

    return correct_LAS, total


def compute_LAS_chuliu(heads_chuliu_pred: Tensor, deprels_pred: Tensor, heads_true: Tensor, deprels_true: Tensor):
    deprels_pred = deprel_pred_for_heads(deprels_pred, heads_true)

    # Ignore predictions for the tokens that don't begin a real word (CLS, subwords, padding, etc.)
    mask = (heads_true != DUMMY_ID)
    total = __sum_2d_tensor(mask)
    correct_head = heads_chuliu_pred[mask] == heads_true[mask]
    correct_deprel = deprels_pred.max(dim=1)[1][mask] == deprels_true[mask]

    correct_LAS = tsum(correct_head & correct_deprel).item()
    LAS_epoch = tsum(correct_head & correct_deprel).item()

    return LAS_epoch, correct_LAS, total


def confusion_matrix(deprels_pred, deprels_true, heads_true, conf_matrix):
    # Ignore predictions for the tokens that don't begin a real word (CLS, subwords, padding, etc.)
    mask = (heads_true != DUMMY_ID)
    deprels_pred = deprel_pred_for_heads(deprels_pred, heads_true)

    trues = deprels_true[mask]
    preds = deprels_pred.max(dim=1)[1][mask]

    for p, t in zip(preds, trues):
        conf_matrix[p, t] += 1

    return conf_matrix
