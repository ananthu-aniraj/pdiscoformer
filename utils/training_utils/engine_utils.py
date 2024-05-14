import torch
from enum import Enum
from .snapshot_class import Snapshot


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter(object):
    """
    Customized progress meter
    Ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))


def load_state_dict_pdisco(snapshot_data):
    """Load state dict of a snapshot.

    Args:
        snapshot_data (dict): dictionary containing the state dict of a snapshot
    """
    snapshot = Snapshot(**snapshot_data)
    state_dict = snapshot.model_state
    return snapshot, state_dict


def change_key(ordered_dict_obj, old, new):
    for _ in range(len(ordered_dict_obj)):
        k, v = ordered_dict_obj.popitem(False)
        ordered_dict_obj[new if old == k else k] = v


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
