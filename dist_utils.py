# Adapted from salesforce@LAVIS. Below is the original copyright:
#  Copyright (c) 2022, salesforce.com, inc.
#  All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#  For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import datetime
import functools

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #     args.gpu = int(os.environ["LOCAL_RANK"])
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # else:
    #     print("Not using distributed mode")
    #     args.use_distributed = False
    #     return

    # args.use_distributed = True


    if args.use_distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        args.rank = args.gpu
        args.world_size = args.ngpus_per_node

        torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"
        print(
            "| distributed init (rank {}, world {}): {}".format(
                args.rank, args.world_size, args.dist_url
            ),
            flush=True,
        )
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(
                days=365
            ),  # allow auto-downloading and de-compressing
        )
        dist.barrier()
        setup_for_distributed(args.rank == 0)

    else:
        print("Not using distributed mode")
        return

def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor