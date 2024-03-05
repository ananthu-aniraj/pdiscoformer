import torch


def orthogonality_loss(all_features, num_landmarks):
    """
    Calculate orthogonality loss for a feature map
    :param all_features: The feature map with shape (batch_size, channels, height, width) re-weighted by the part probability attention map
    :param num_landmarks: Number of landmarks/parts
    :return:
    """
    normed_feature = torch.nn.functional.normalize(all_features, dim=1)
    similarity_fg = torch.matmul(normed_feature.permute(0, 2, 1).contiguous(), normed_feature)
    similarity_fg = torch.sub(similarity_fg, torch.eye(num_landmarks + 1, device=all_features.device))
    orth_loss = torch.mean(torch.square(similarity_fg))
    return orth_loss


def orthogonality_loss_modulation(modulation_vector):
    """
    Calculate orthogonality loss for the modulation vector
    :param modulation_vector: The modulation vector with shape (1, feature_dim, num_landmarks)
    :return:
    """
    normed_modulation_vector = (modulation_vector / modulation_vector.norm(dim=1, keepdim=True)).squeeze(0)
    similarity_mat = torch.matmul(normed_modulation_vector.permute(1, 0).contiguous(), normed_modulation_vector)
    similarity_mat = torch.sub(similarity_mat, torch.eye(similarity_mat.shape[0], device=modulation_vector.device))
    orthogonality_loss_mod = torch.mean(torch.square(similarity_mat))
    return orthogonality_loss_mod
