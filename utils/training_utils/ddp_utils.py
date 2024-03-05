import os
import torch
from torch.distributed import init_process_group
import torch.distributed as dist
import numpy as np
import subprocess
import socket


def ddp_setup():
    is_slurm_job = "SLURM_NODEID" in os.environ
    if is_slurm_job:
        # Define the process group based on SLURM env variables
        # number of nodes / node ID
        n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
        n_gpu_per_node = world_size // n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)

        # define whether this is the master process / if we are in distributed mode
        is_master = node_id == 0 and local_rank == 0
        multi_node = n_nodes > 1
        multi_gpu = world_size > 1

        # summary
        prefix = "%i - " % global_rank
        print(prefix + "Number of nodes: %i" % n_nodes)
        print(prefix + "Node ID        : %i" % node_id)
        print(prefix + "Local rank     : %i" % local_rank)
        print(prefix + "Global rank    : %i" % global_rank)
        print(prefix + "World size     : %i" % world_size)
        print(prefix + "GPUs per node  : %i" % n_gpu_per_node)
        print(prefix + "Master         : %s" % str(is_master))
        print(prefix + "Multi-node     : %s" % str(multi_node))
        print(prefix + "Multi-GPU      : %s" % str(multi_gpu))
        print(prefix + "Hostname       : %s" % socket.gethostname())
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
    print("Initializing PyTorch distributed ...")
    init_process_group(init_method='env://', backend="nccl")
    torch.cuda.set_device(local_rank)
    return


def set_seeds(seed_value: int = 42):
    # Set the manual seeds
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)


def reduce_tensor(tensor: torch.Tensor, world_size: int):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def multi_gpu_check():
    """
    Check if there are multiple GPUs available for DDP
    :return:
    use_ddp: bool, whether to use DDP or not
    """
    if torch.cuda.device_count() > 1:
        use_ddp = True
    else:
        use_ddp = False
    return use_ddp


def calculate_effective_batch_size(args):
    """
    Calculate the effective batch size for DDP
    :param args: Arguments from the argument parser
    :return:
    effective_batch_size: int, effective batch size
    """
    batch_size = args.batch_size
    use_ddp = multi_gpu_check()
    is_slurm_job = "SLURM_NODEID" in os.environ
    if is_slurm_job:
        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        if use_ddp:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = 1

    effective_batch_size = batch_size * world_size
    print(f'Effective batch size: {effective_batch_size}')
    return effective_batch_size
