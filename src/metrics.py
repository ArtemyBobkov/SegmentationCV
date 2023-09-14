import torch


def DiffFgDICE(pred, label):

    eps = 1e-6

    pred_max = pred.max() if pred.max() != 0 else 1
    label_max = label.max() if label.max() != 0 else 1
    pred = torch.nn.ReLU()(pred / pred_max - 0.5)
    label = torch.nn.ReLU()(label / label_max - 0.5)
    intersection = (pred * label).sum()

    true_pos = torch.sum(label * pred)
    false_pos = torch.sum((1 - label) * pred)
    false_neg = torch.sum(label * (1 - pred))

    return (2. * intersection + eps) / (2 * true_pos + false_pos + false_neg + eps) / 0.5


def DiffFgMSE(pred, label):
    return 1 / len(label) * ((label - pred) ** 2).sum()
