"""
Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py and https://github.com/google-research/syn-rep-learn/blob/45f451b0d53d25eecdb4d7b9e5a852e1c43e7f5b/StableRep/models/losses.py#L49
"""
import torch
import torch.nn as nn
import torch.distributed.nn
from utils.training_utils.ddp_utils import get_rank, is_dist_avail_and_initialized, concat_all_gather
from utils.misc_utils import compute_cross_entropy


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


class MultiPosConLoss(nn.Module):
    """Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    Adapted from https://github.com/google-research/syn-rep-learn/blob/45f451b0d53d25eecdb4d7b9e5a852e1c43e7f5b/StableRep/models/losses.py#L49
    """

    def __init__(self, temperature=0.07):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feature_vectors, local_rank: int = 0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            feature_vectors: hidden vector of shape [bsz, embedding_size, num_prototypes]
            local_rank: Local rank of the current process
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda:%d' % local_rank)
                  if torch.cuda.is_available() else torch.device('cpu'))

        if len(feature_vectors.shape) < 3 or len(feature_vectors.shape) > 4:
            raise ValueError('`features` needs to be [bsz, embedding_size, num_prototypes] '
                             'exactly 3 dimensions are required')

        batch_size = feature_vectors.shape[0]
        embedding_size = feature_vectors.shape[1]
        num_prototypes = feature_vectors.shape[2]

        #  Normalize the feature vectors
        norm_features = torch.nn.functional.normalize(feature_vectors, dim=1,
                                                      p=2)  # (bsz, embedding_size, num_prototypes)
        # Create a feature matrix with shape (bsz*num_prototypes, embedding_size) so that each row is a prototype vector
        feats = norm_features.permute(0, 2, 1).contiguous().view(-1,
                                                                 embedding_size)  # (bsz*num_prototypes, embedding_size)
        labels = torch.arange(num_prototypes, device=local_rank).repeat(batch_size).contiguous().view(-1,
                                                                                                      1)  # (bsz*num_prototypes, 1)
        local_batch_size = feats.size(0)

        if is_dist_avail_and_initialized():
            all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
            all_labels = concat_all_gather(labels)  # no gradient gather
        else:
            all_feats = feats
            all_labels = labels

        # Create label matrix, since in our specific case the
        # label matrix in side each batch is the same, so
        # we can just create it once and reuse it. For other
        # cases, user need to compute it for each batch
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device) +
                local_batch_size * get_rank(),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask
        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9
        # optional: minus the largest logit to stabilize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)
        return loss
