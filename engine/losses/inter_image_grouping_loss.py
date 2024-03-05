"""
Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py and https://github.com/google-research/syn-rep-learn/blob/45f451b0d53d25eecdb4d7b9e5a852e1c43e7f5b/StableRep/models/losses.py#L49
"""
import torch


def inter_image_grouping_loss(feature_vectors):
    """
        Args:
        feature_vectors: hidden vector of shape [bsz, embedding_size, num_prototypes]
    Returns:
        A loss scalar.
    """

    if len(feature_vectors.shape) < 3 or len(feature_vectors.shape) > 4:
        raise ValueError('`features` needs to be [bsz, embedding_size, num_prototypes] '
                         'exactly 3 dimensions are required')

    #  Normalize the feature vectors
    norm_features = torch.nn.functional.normalize(feature_vectors, dim=1,
                                                  p=2)  # (bsz, embedding_size, num_prototypes)

    norm_features_batch = norm_features.permute(2, 1, 0).contiguous()  # (num_prototypes, embedding_size, bsz)
    similarity_batch = torch.matmul(norm_features_batch.permute(0, 2, 1).contiguous(),
                                    norm_features_batch)  # (num_prototypes, bsz, bsz)
    inter_image_loss = torch.mean(torch.square(similarity_batch))

    # Maximize similarity between inter-batch prototypes
    loss = 1 - inter_image_loss

    return loss
