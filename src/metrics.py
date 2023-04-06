def DiffFgDICE(pred, label):
    mask = label != 0
    true_pos = (mask * pred).sum()
    false_pos = (~mask * pred).sum()
    false_neg = (mask * (label - pred)).sum()
    return (2 * true_pos) / (2 * true_pos + false_pos + false_neg)


def DiffFgMSE(pred, label):
    return 1 / len(label) * ((label - pred) ** 2).sum()
