import torch


def orthogonality_loss(all_features, num_landmarks):
    """
    Calculate orthogonality loss for a feature map
    Ref: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L44
    :param all_features: The feature map with shape (batch_size, channels, height, width) re-weighted by the part probability attention map
    :param num_landmarks: Number of landmarks/parts
    :return:
    """
    normed_feature = torch.nn.functional.normalize(all_features, dim=1)
    similarity_fg = torch.matmul(normed_feature.permute(0, 2, 1).contiguous(), normed_feature)
    similarity_fg = torch.sub(similarity_fg, torch.eye(num_landmarks + 1, device=all_features.device))
    orth_loss = torch.mean(torch.square(similarity_fg))
    return orth_loss
