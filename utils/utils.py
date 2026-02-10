import torch


def compute_class_weights(dataset):
    labels = [data.y.item() for data in dataset]
    counts = torch.bincount(torch.tensor(labels))
    num_classes = len(counts)

    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * num_classes
    return weights
