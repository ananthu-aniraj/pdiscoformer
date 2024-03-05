# Implementation of the Hungarian loss function.
import torch
from scipy.optimize import linear_sum_assignment


def instance_matcher_nll(per_instance_scores, targets):
    """
    Find the instances with the lowest loss for each target.
    Inspired by https://github.com/facebookresearch/detr/blob/main/models/matcher.py
    Negative log-likelihood loss is used as the cost function.
    :param per_instance_scores: vector of logits for each instance of shape (batch_size, num_instances, num_classes)
    :param targets: vector of target indices of shape (batch_size)
    :return:
    outputs: vector of logits for each instance of shape (batch_size, num_classes)
    loss_instances: Mean loss across all instances
    """
    num_instances = per_instance_scores.shape[1]
    batch_size = per_instance_scores.shape[0]
    per_inst_probs = per_instance_scores.flatten(0, 1).contiguous().softmax(dim=-1)  # (batch_size * num_instances, num_classes)

    with torch.no_grad():
        tgt_ids = targets.unsqueeze(1)  # (batch_size, 1)
        cost_class = -per_inst_probs[:, tgt_ids]  # (batch_size * num_instances, batch_size, 1)
        cost_class = cost_class.view(batch_size, num_instances, -1).cpu()  # (batch_size, num_instances, batch_size)
        sizes = [1 for v in targets]  # Assume 1 instance per image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_class.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        selected_instances = torch.cat([src for (src, _) in indices])
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])

    outputs = per_instance_scores[batch_idx, selected_instances, :]
    targets_exp = targets.unsqueeze(1).expand(batch_size, num_instances)
    cost_matrix = torch.nn.functional.cross_entropy(per_instance_scores.permute(0, 2, 1), targets_exp,
                                                    reduction='none')  # (batch_size, num_instances)

    loss_instances = cost_matrix.min(dim=0).values.mean()
    return outputs, loss_instances


def instance_matcher_cr(per_instance_scores, targets):
    """
    Find the instances with the lowest loss for each target.
    Cross entropy loss is used as the cost function.
    :param per_instance_scores: vector of logits for each instance of shape (batch_size, num_instances, num_classes)
    :param targets: Target indices of shape (batch_size)
    :return:
    outputs: Logits for each instance of shape (batch_size, num_classes)
    loss_instances: Mean loss across all instances
    """
    num_instances = per_instance_scores.shape[1]
    batch_size = per_instance_scores.shape[0]
    targets_exp = targets.unsqueeze(1).expand(batch_size, num_instances)
    cost_class = torch.nn.functional.cross_entropy(per_instance_scores.permute(0, 2, 1).contiguous(), targets_exp,
                                                   reduction='none')  # (batch_size, num_instances)
    loss_instances = cost_class.min(dim=0).values.mean()
    # Pick the instance with the lowest loss for each image in the batch
    selected_instances = cost_class.argmin(dim=1)
    outputs = per_instance_scores[torch.arange(batch_size), selected_instances, :]
    return outputs, loss_instances, selected_instances

