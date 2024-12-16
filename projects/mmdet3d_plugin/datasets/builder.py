import copy
import platform
import random
from functools import partial

import numpy as np
from mmengine.dataset.utils import default_collate as collate
from mmengine.dist import get_dist_info
from mmengine.registry import build_from_cfg, Registry
from torch.utils.data import DataLoader
from mmdet.registry import DATASETS

from projects.mmdet3d_plugin.datasets.samplers import (
    GroupInBatchSampler,
    DistributedGroupSampler,
    DistributedSampler,
    build_sampler
)

from torch.utils.data import Sampler, RandomSampler


def _concat_dataset(cfg, default_args=None):
    from mmengine.dataset.dataset_wrapper import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_from_cfg(data_cfg, DATASETS, default_args))

    return ConcatDataset(datasets, separate_eval)

    return dataset

class GroupSampler(Sampler):
    """
    A minimal custom implementation of GroupSampler for distributed group sampling.
    """
    def __init__(self, data_source, num_samples_per_group):
        """
        Args:
            data_source (Dataset): Dataset from which indices will be sampled.
            num_samples_per_group (int): Number of indices to sample per group.
        """
        self.data_source = data_source
        self.num_samples_per_group = num_samples_per_group
        self.num_samples = len(data_source)

    def __iter__(self):
        """
        Sampler iterator that performs random sampling within groups.
        """
        # Shuffle the indices randomly
        indices = list(range(self.num_samples))
        random.shuffle(indices)

        # Yield indices in groups of `num_samples_per_group`
        for i in range(0, len(indices), self.num_samples_per_group):
            yield indices[i:i + self.num_samples_per_group]

    def __len__(self):
        """
        Returns the number of groups (total length divided by num_samples_per_group).
        """
        return (len(self.data_source) + self.num_samples_per_group - 1) // self.num_samples_per_group


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    shuffler_sampler=None,
    nonshuffler_sampler=None,
    runner_type="EpochBasedRunner",
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    batch_sampler = None
    if runner_type == 'IterBasedRunner':
        print("Use GroupInBatchSampler !!!")
        sampler = RandomSampler(dataset)
        batch_sampler = GroupInBatchSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed,
            sampler=sampler,
        )
        batch_size = 1
        sampler = None
        num_workers = workers_per_gpu
    elif dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            print("Use DistributedGroupSampler !!!")
            sampler = build_sampler(
                shuffler_sampler
                if shuffler_sampler is not None
                else dict(type="DistributedGroupSampler"),
                dict(
                    dataset=dataset,
                    samples_per_gpu=samples_per_gpu,
                    num_replicas=world_size,
                    rank=rank,
                    seed=seed,
                ),
            )
        else:
            sampler = build_sampler(
                nonshuffler_sampler
                if nonshuffler_sampler is not None
                else dict(type="DistributedSampler"),
                dict(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed,
                ),
            )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print("WARNING!!!!, Only can be used for obtain inference speed!!!!")
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmengine.registry  import Registry, build_from_cfg, DATASETS



if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")


def custom_build_dataset(cfg, default_args=None):
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    except:
        CBGSDataset = None
    from mmengine.dataset import ConcatDataset, RepeatDataset, ClassBalancedDataset


    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg]
        )
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            custom_build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg["dataset"], default_args),
            cfg["oversample_thr"],
        )
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(
            custom_build_dataset(cfg["dataset"], default_args)
        )
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
