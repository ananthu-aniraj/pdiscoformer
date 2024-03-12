# Code for the Equivariance Loss

import torch
from utils.data_utils.reversible_affine_transform import rigid_transform


def equivariance_loss(maps, equiv_maps, source, num_landmarks, translate, angle, scale, shear=0.0):
    """
    This function calculates the equivariance loss
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L67
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :param equiv_maps: Attention maps for same images after an affine transformation and then passed through the model
    :param source: Original mini-batch of images
    :param num_landmarks: Number of landmarks/parts
    :param translate: Translation parameters for the affine transformation
    :param angle: Angle parameter for the affine transformation
    :param scale: Scale parameter for the affine transformation
    :param shear: Shear parameter for the affine transformation
    :return:
    """

    translate = [(t * maps.shape[-1] / source.shape[-1]) for t in translate]
    rot_back = rigid_transform(img=equiv_maps, angle=angle, translate=translate,
                               scale=scale, shear=shear, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :],
                                       (-1, num_landmarks,
                                        num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :],
                                         (-1, num_landmarks,
                                          num_elements_per_map))
    cos_sim_equiv = torch.nn.functional.cosine_similarity(orig_attmap_vector,
                                                          transf_attmap_vector, -1)
    loss_equiv = (1 - torch.mean(cos_sim_equiv))

    return loss_equiv
