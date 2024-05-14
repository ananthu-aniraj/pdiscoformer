import torch
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def plot_iou_figs(iou_values, iou_values_bg, model_path):
    plt.figure()
    plt.plot(iou_values)
    plt.xlabel('Batch')
    plt.ylabel('IoU')
    plt.title('Foreground IoU')
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(model_path), 'fg_iou.png'), bbox_inches='tight')
    plt.figure()
    plt.plot(iou_values_bg)
    plt.xlabel('Batch')
    plt.ylabel('IoU')
    plt.title('Background IoU')
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(model_path), 'bg_iou.png'), bbox_inches='tight')
    plt.close()


class FgBgIoU:
    """
    Class to calculate the IoU for the foreground and background classes
    """

    def __init__(self, model, data_loader, device):
        """
        Initialize the class
        :param device: Device
        :param model: Model
        """
        self.metric_fg = BinaryJaccardIndex().to(device, non_blocking=True)
        self.metric_bg = BinaryJaccardIndex().to(device, non_blocking=True)
        self.model = model
        self.device = device
        self.num_parts = model.num_landmarks
        self.data_loader = data_loader

    def calculate_iou(self, model_path):
        """
        Function to calculate the IoU for the foreground class
        :return: Foreground IoU
        """
        iou_values = []
        iou_values_bg = []
        self.metric_fg.reset()
        self.metric_bg.reset()
        self.model.eval()
        for (img, _, mask) in tqdm(self.data_loader, desc='Testing'):
            img = img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with torch.inference_mode():
                assign = self.model(img)[1]

            map_argmax = torch.nn.functional.interpolate(assign, size=(mask.shape[-2], mask.shape[-1]),
                                                         mode='bilinear',
                                                         align_corners=True).argmax(dim=1)

            map_argmax[map_argmax != self.num_parts] = 1
            map_argmax[map_argmax == self.num_parts] = 0
            mask = mask.float()
            map_argmax = map_argmax.float()
            inv_mask = 1 - mask
            inv_map_argmax = 1 - map_argmax
            iou = self.metric_fg(map_argmax, mask)
            iou_values.append(iou.item())
            iou_bg = self.metric_bg(inv_map_argmax, inv_mask)
            iou_values_bg.append(iou_bg.item())
        plot_iou_figs(iou_values, iou_values_bg, model_path)

