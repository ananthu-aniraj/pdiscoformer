import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc
import numpy as np
import skimage
from pathlib import Path
import os
import torch

from utils.data_utils.transform_utils import inverse_normalize_w_resize
from utils.misc_utils import factors

# Define the colors to use for the attention maps
colors = cc.glasbey_category10


class VisualizeAttentionMaps:
    def __init__(self, snapshot_dir="", save_resolution=(256, 256), alpha=0.5, sub_path_test="",
                 dataset_name="", bg_label=0, batch_size=32, num_parts=15, plot_ims_separately=False,
                 plot_landmark_amaps=False):
        """
        Plot attention maps and optionally landmark centroids on images.
        :param snapshot_dir: Directory to save the visualization results
        :param save_resolution: Size of the images to save
        :param alpha: The transparency of the attention maps
        :param sub_path_test: The sub-path of the test dataset
        :param dataset_name: The name of the dataset
        :param bg_label: The background label index in the attention maps
        :param batch_size: The batch size
        :param num_parts: The number of parts in the attention maps
        :param plot_ims_separately: Whether to plot the images separately
        :param plot_landmark_amaps: Whether to plot the landmark attention maps
        """
        self.save_resolution = save_resolution
        self.alpha = alpha
        self.sub_path_test = sub_path_test
        self.dataset_name = dataset_name
        self.bg_label = bg_label
        self.snapshot_dir = snapshot_dir
        if self.snapshot_dir == "":
            matplotlib.use('Qt5Agg')
        self.resize_unnorm = inverse_normalize_w_resize(resize_resolution=self.save_resolution)
        self.batch_size = batch_size
        self.nrows = factors(self.batch_size)[-1]
        self.ncols = factors(self.batch_size)[-2]
        self.num_parts = num_parts
        self.req_colors = colors[:num_parts]
        self.plot_ims_separately = plot_ims_separately
        self.plot_landmark_amaps = plot_landmark_amaps
        if self.nrows == 1 and self.ncols == 1:
            self.figs_size = (10, 10)
        else:
            self.figs_size = (self.ncols * 2, self.nrows * 2)

    def recalculate_nrows_ncols(self):
        self.nrows = factors(self.batch_size)[-1]
        self.ncols = factors(self.batch_size)[-2]
        if self.nrows == 1 and self.ncols == 1:
            self.figs_size = (10, 10)
        else:
            self.figs_size = (self.ncols * 2, self.nrows * 2)

    @torch.no_grad()
    def show_maps(self, ims, maps, epoch=0, curr_iter=0, extra_info=""):
        """
        Plot images, attention maps and landmark centroids.
        Parameters
        ----------
        ims: Tensor, [batch_size, 3, width_im, height_im]
            Input images on which to show the attention maps
        maps: Tensor, [batch_size, number of parts + 1, width_map, height_map]
            The attention maps to display
        epoch: int
            The epoch number
        curr_iter: int
            The current iteration number
        extra_info: str
            Any extra information to add to the file name
        """
        ims = self.resize_unnorm(ims)
        if ims.shape[0] != self.batch_size:
            self.batch_size = ims.shape[0]
            self.recalculate_nrows_ncols()
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, squeeze=False, figsize=self.figs_size)
        ims = (ims.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        map_argmax = torch.nn.functional.interpolate(maps.clone().detach(), size=self.save_resolution,
                                                     mode='bilinear',
                                                     align_corners=True).argmax(dim=1).cpu().numpy()
        for i, ax in enumerate(axs.ravel()):
            curr_map = skimage.color.label2rgb(label=map_argmax[i], image=ims[i], colors=self.req_colors,
                                               bg_label=self.bg_label, alpha=self.alpha)
            ax.imshow(curr_map)
            ax.axis('off')
        save_dir = Path(os.path.join(self.snapshot_dir, 'results_vis_' + self.sub_path_test))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f'{epoch}_{curr_iter}_{self.dataset_name}{extra_info}.png')
        fig.tight_layout()
        if self.snapshot_dir != "":
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close('all')

        if self.plot_ims_separately:
            fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, squeeze=False, figsize=self.figs_size)
            for i, ax in enumerate(axs.ravel()):
                ax.imshow(ims[i])
                ax.axis('off')
            save_path = os.path.join(save_dir, f'image_{epoch}_{curr_iter}_{self.dataset_name}{extra_info}.jpg')
            fig.tight_layout()
            if self.snapshot_dir != "":
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            else:
                plt.show()
        plt.close('all')

        if self.plot_landmark_amaps:
            if self.batch_size > 1:
                raise ValueError('Not implemented for batch size > 1')
            for i in range(self.num_parts):
                fig, ax = plt.subplots(1, 1, figsize=self.figs_size)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(maps[0, i, ...].detach().cpu().numpy(), cmap='cet_gouldian')
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.axis('off')
                save_path = os.path.join(save_dir,
                                         f'landmark_{i}_{epoch}_{curr_iter}_{self.dataset_name}{extra_info}.png')
                fig.tight_layout()
                if self.snapshot_dir != "":
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                else:
                    plt.show()
                plt.close()

        plt.close('all')
