""" Nequip.train.trainer

Todo:

isolate the loss function from the training procedure
enable wandb resume
make an interface with ray

"""
import sys
import inspect
import logging
import os
import random
import uuid
from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Callable, Optional, Union, Tuple, List
from pathlib import Path
import json

if sys.version_info[1] >= 7:
    import contextlib
else:
    # has backport of nullcontext
    import contextlib2 as contextlib

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage

import nequip.utils.optim
from nequip.data import DataLoader, AtomicData, AtomicDataDict, AtomicDataset
from nequip.utils import (
    Output,
    Config,
    instantiate_from_cls_name,
    instantiate,
    save_file,
    load_file,
    load_callable,
    atomic_write,
    finish_all_writes,
    atomic_write_group,
)
from nequip.utils.versions import check_code_version
from nequip.model import model_from_config
from nequip.utils.compile_utils import training_autocast_context

from nequip.nn import tune_model_coeff_nosoc, tune_model_ztype

from .loss import Loss, LossStat, DistributedLossStat, distributed_objective
from .e2ip import e2IPLoss
from .metrics import Metrics
from ._param_groups import (
    build_optimizer_param_groups,
    check_param_group_options,
    describe_structure_mismatch,
    resolve_param_group_index,
)
from ._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION
from .early_stopping import EarlyStopping

_DDP_WARMUP_DONE = False


def warmup_process_group_for_ddp() -> None:
    """Run a one-time warmup collective before ``DistributedDataParallel``.

    torch 2.11.0+cu128 / NCCL 2.28.9 SIGSEGVs inside DDP construction
    (coalesced broadcast on a fresh communicator) on both RTX 5090 and 4090;
    a single tiny broadcast on the communicator before DDP construction avoids
    it deterministically (probe jobs 2290284/2290286 crash vs 2290293/2290294
    pass). Idempotent per process. Non-NCCL backends broadcast a CPU tensor so
    gloo CPU tests never touch CUDA.
    """
    global _DDP_WARMUP_DONE
    if _DDP_WARMUP_DONE:
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        tensor = torch.zeros(
            1, device=torch.device("cuda", torch.cuda.current_device())
        )
    else:
        tensor = torch.zeros(1)
    dist.broadcast(tensor, src=0)
    _DDP_WARMUP_DONE = True


# recovery schema version for the canonical + rank-local checkpoint contract
_RECOVERY_SCHEMA_VERSION = 1
# name of the per-rank RNG state file next to the canonical trainer file
_RANK_STATE_FILE_PATTERN = "trainer-rank{rank}.pth"


def _ddp_broadcast_object(obj) -> None:
    """Broadcast a python object list from rank 0 to every rank."""

    kwargs = {}
    if dist.get_backend() != "gloo" and torch.cuda.is_available():
        kwargs["device"] = torch.device("cuda", torch.cuda.current_device())
    dist.broadcast_object_list(obj, src=0, **kwargs)


class Trainer:
    """Class to train a model to minimize forces

    This class isolate the logging and callback functions from the training procedure.
    The class instance can be easily save/load for restart;
    There are options to append the old logging file or to restart a new file,
    and options to append old model files or to save in a new path.

    Examples:

    To start a training.

    '''python
    trainer = Trainer(model, learning_rate=1e-2)
    trainer.set_dataset(dataset)
    trainer.train()
    '''

    To load and restart a training.

    '''python
    trainer = Trainer.from_file(filename)
    trainer.set_dataset(dataset)
    trainer.train()
    '''

    To load, slightly revise content and then resume training
    '''python
    dictionary = nequip.utils.savenload.load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=filename,
            enforced_format="torch",
            map_location="cpu",
        )
    dictionary["progress"]["stop_arg"] = None
    dictionary["max_epochs"] += 100
    trainer = Trainer.from_dict(
        dictionary, append=False
    )
    trainer.train()
    '''

    For a fresh run, the simulation results will be stored in the 'root/run_name' folder. With
    - "log" : plain text information about the whole training process
    - "metrics_epoch.csv" : txt matrice format (readable by np.loadtxt) with loss/mae from training and validation of each epoch
    - "metrics_batch.csv" : txt matrice format (readable by np.loadtxt) with training loss/mae of each batch
    - "best_model.pth": the best model. save the model when validation mae goes lower
    - "last_model.pth": the last model. save the model every log_epoch_freq epoch
    - "trainer_save.pth": all the training information. The file used for loading and restart

    For restart run, the default set up is to not append to the original folders and files.
    The Output class will automatically build a folder call root/run_name
    If append mode is on, the log file will be appended and the best model and last model will be overwritten.

    More examples can be found in tests/train/test_trainer.py

    Note:
        The data in the dataloader has to be DataLoader and Data in torch_geometric.data

        Only optimizers from torch.optim and schedulers from torch.optim.lr_scheduler
        are supported

        Extra arguments needed by optimizer and scheduler can be directly taken in as
        long as they are exactly the same as the ones listed in the init functions. To
        avoid ambituity, one can also address the argument with "lr_" and "optim_" prefixes
        for the lr_scheduler and optimizer, correspondingly.
        Or save them in a dictionary as "optim_params" and "lr_scheduler_params"

        The priority of parameters for optimizers/schedulers is:

            key < optim_key < optim_params[key]

    For developer,

    - all parameters added to the init function arguments will be automatically
      saved as class attribute with the same name, and saved to the pth file
    - To store and recover new additional attributes, it needs explicit statement in the as_dict and from_dict
    - To check which parameters were actually used for optimizer, scheduler and loaders initialization, please use verbose=debug.
      The screen output will list all the details


    Args:
        model: neural network model

        seed (int): random seed number
        dataset_seed (int): random seed for dataset operations

        loss_coeffs (dict): dictionary to store coefficient and loss functions

        max_epochs (int): maximum number of epochs

        learning_rate (float): initial learning rate
        lr_scheduler_name (str): scheduler name
        lr_scheduler_kwargs (dict): parameters to initialize the scheduler

        optimizer_name (str): name for optimizer
        optim_kwargs (dict): parameters to initialize the optimizer
        optimizer_param_groups (list, optional): ordered named parameter groups,
            each ``{name, match, <optimizer options such as lr, weight_decay>}``.
            ``match`` lists parameter-name prefixes or ``"re:<regex>"`` patterns
            (``re.search``); the first matching group wins and one group without
            ``match`` is the catch-all. Names are the unwrapped model's
            ``named_parameters()`` (state_dict keys, no DDP ``module.`` prefix).
            Only parameters trainable when the optimizer is built are grouped
            (parameters unfrozen later are not optimized; the spin/ztype ladders
            are rejected); each must land in a group and each group must be
            non-empty. The list order is the optimizer group order and is checked
            on resume. Each group's learning rate is logged as an extra
            ``LR_<name>`` column of the epoch metrics (also usable in
            ``early_stopping_lower_bounds``). Default ``None``: a single group
            with all model parameters.
        early_stopping_lr_group (int or str): optimizer param group (index or
            ``optimizer_param_groups`` name) whose learning rate is reported as
            ``LR`` and therefore used by ``early_stopping_lower_bounds: {LR: ...}``.
            Defaults to group 0, so ``LR`` is unchanged by default.

        batch_size (int): size of each batch
        validation_batch_size (int): batch size for evaluating the model for validation
        shuffle (bool): parameters for dataloader
        n_train (int): # of frames for training
        n_val (int): # of frames for validation
        exclude_keys (list):  fields from dataset to ignore.
        dataloader_num_workers (int): `num_workers` for the `DataLoader`s
        train_idcs (optional, list):  list of frames to use for training
        val_idcs (list):  list of frames to use for validation
        train_val_split (str):  "random" or "sequential"

        init_callbacks (list): list of callback function at the begining of the training
        start_of_epoch_callbacks (list): callables (or dotted import paths)
            called as ``callback(trainer)`` on every rank at the start of every
            training epoch, before its first batch. ``trainer.iepoch`` is the
            0-based index of the epoch about to run. The initial validation
            (``iepoch == -1``) is not a training epoch. A restart makes no
            separate call: loading a checkpoint does not invoke these callbacks,
            and the first call of a resumed run is at the start of the first
            resumed epoch (``iepoch`` = number of completed epochs), after all
            model, optimizer, EMA, scheduler and RNG state has been restored.
            Callbacks should therefore derive their state from
            ``trainer.iepoch`` rather than count calls. Under native DDP (the
            default distributed implementation) ``trainer.model`` is the
            ``DistributedDataParallel`` wrapper; use ``trainer.model.module``
            (or ``getattr(trainer.model, "module", trainer.model)``, which also
            covers single-process and ``SPINGNN_DDP_IMPL=manual`` runs) to reach
            the model's attributes and buffers.
        end_of_epoch_callbacks (list): list of callback functions at the end of each epoch
        end_of_batch_callbacks (list): list of callback functions at the end of each batch
        end_of_train_callbacks (list): list of callback functions between traing/validation
        final_callbacks (list): list of callback functions at the end of the training

        log_batch_freq (int): frequency to log at the end of a batch
        batch_metrics_log_interval (int): materialize and write batch metrics
            every N batches (and always on the final batch); default 1 keeps
            the historical per-batch CSV/log output. Epoch accumulators are
            updated on every batch regardless of this setting.
        ddp_broadcast_buffers (bool, optional): native DDP buffer broadcast;
            ``None`` preserves the PyTorch default (True).
        ddp_static_graph (bool, optional): native DDP static-graph optimization;
            requires the same parameter usage on every iteration.
        ddp_gradient_as_bucket_view (bool, optional): native DDP gradient bucket
            views; ``None`` preserves the PyTorch default (False).
        ddp_find_unused_parameters (bool, optional): native DDP unused-parameter
            detection. When absent, ``SPINGNN_DDP_FIND_UNUSED`` is consulted
            (default enabled), preserving the existing precedence.
            Disabling buffer broadcasts requires model buffers to be identical
            on all ranks or updated identically on every rank. Static-graph
            mode requires the same set of used parameters on every iteration.
        log_epoch_freq (int): frequency to save at the end of an epoch
        save_checkpoint_freq (int): frequency to save the intermediate checkpoint. no saving when the value is not positive.
        save_ema_checkpoint_freq (int): frequency to save the intermediate ema checkpoint. no saving when the value is not positive.

        verbose (str): verbosity level, i.e. "INFO", "WARNING", "DEBUG". case insensitive

    Additional Attributes:

        init_keys (list): list of parameters needed to reconstruct this instance
        dl_train (DataLoader): training data
        dl_val (DataLoader): test data
        iepoch (int): # of epoches ran
        stop_arg (str): reason why the training stops
        batch_mae (float): the mae of the latest batch
        mae_dict (dict): all loss, mae of the latest validation
        best_metrics (float): current best validation mae
        best_epoch (float): current best epoch
        best_model_path (str): path to save the best model
        last_model_path (str): path to save the latest model
        trainer_save_path (str): path to save the trainer.
             Default is trainer.(date).pth at the current folder


    The pseudocode of the workflow and location of the callback functions

    ```
    init():
        initialize optimizer, schduler and loss function

    train():
       init model
       init_callbacks
       while (not stop):
            start_of_epoch_callbacks (training epochs only)
            training batches
                end_of_batch_callbacks
            end_of_train_callbacks
            validation_batches
            end_of_epoch_callbacks
       final_callbacks
    ```
    """

    stop_keys = ["max_epochs", "early_stopping", "early_stopping_kwargs"]
    object_keys = ["lr_sched", "optim", "ema", "early_stopping_conds"]
    lr_scheduler_module = torch.optim.lr_scheduler
    optim_module = torch.optim

    def __init__(
        self,
        model,
        model_builders: Optional[list] = [],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: Optional[int] = None,
        dataset_seed: Optional[int] = None,
        loss_coeffs: Union[dict, str] = AtomicDataDict.TOTAL_ENERGY_KEY,
        validation_loss_coeffs: Optional[Union[dict, str, List[str]]] = None,
        use_e2IP_loss: bool = False,
        validation_use_e2IP_loss: bool = False,
        evidential_key: str = "forces",
        loss_reg: float = 0.1,
        loss_epsilon: float = 1e-6,
        e2ip_spin_force_scale: float = 40.0,
        e2ip_likelihood_scale: float = 1.0,
        e2ip_likelihood_dtype: str = "float32",
        ldetach_method: bool = False,
        lauto_loss: bool = False,
        train_on_keys: Optional[List[str]] = None,
        metrics_components: Optional[Union[dict, str]] = None,
        metrics_key: str = f"{VALIDATION}_" + ABBREV.get(LOSS_KEY, LOSS_KEY),
        early_stopping_conds: Optional[EarlyStopping] = None,
        early_stopping: Optional[Callable] = None,
        early_stopping_kwargs: Optional[dict] = None,
        max_epochs: int = 1000000,
        learning_rate: float = 1e-2,
        learning_rate_fine_tune: float = 1e-2,
        lr_scheduler_name: str = "none",
        lr_scheduler_kwargs: Optional[dict] = None,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        optimizer_param_groups: Optional[List[dict]] = None,
        early_stopping_lr_group: Union[int, str] = 0,
        max_gradient_norm: float = float("inf"),
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_use_num_updates=True,
        exclude_keys: list = [],
        batch_size: int = 5,
        validation_batch_size: int = 5,
        shuffle: bool = True,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        dataloader_num_workers: int = 0,
        train_idcs: Optional[list] = None,
        val_idcs: Optional[list] = None,
        train_val_split: str = "random",
        init_callbacks: list = [],
        start_of_epoch_callbacks: list = [],
        end_of_epoch_callbacks: list = [],
        end_of_batch_callbacks: list = [],
        end_of_train_callbacks: list = [],
        final_callbacks: list = [],
        log_batch_freq: int = 100,
        batch_metrics_log_interval: int = 1,
        ddp_broadcast_buffers: Optional[bool] = None,
        ddp_static_graph: Optional[bool] = None,
        ddp_gradient_as_bucket_view: Optional[bool] = None,
        ddp_find_unused_parameters: Optional[bool] = None,
        log_epoch_freq: int = 1,
        save_checkpoint_freq: int = -1,
        save_ema_checkpoint_freq: int = -1,
        report_init_validation: bool = True,
        verbose="INFO",
        train_spin_ladder: bool = False,
        train_spin_ladder_list: Optional[list] = None,
        train_spin_ladder_only_frozen: bool = False,
        train_ztype_eng_ladder: bool = False,
        train_ztype_eng_ladder_full: bool = False,
        lhb: bool = False,
        lnosoc_layer: bool = False,
        lsoc_layer: bool = False,
        **kwargs,
    ):
        self._initialized = False
        self._ddp_enabled = dist.is_available() and dist.is_initialized()
        self._ddp_rank = dist.get_rank() if self._ddp_enabled else 0
        self._ddp_world_size = dist.get_world_size() if self._ddp_enabled else 1
        self._ddp_train_sampler = None
        self._ddp_val_sampler = None
        self._ddp_val_indices = None
        self._ddp_train_padding = 0
        self._ddp_impl = (
            os.environ.get("SPINGNN_DDP_IMPL", "ddp") if self._ddp_enabled else "none"
        )
        if self._ddp_enabled and self._ddp_impl not in {"manual", "ddp"}:
            raise ValueError("SPINGNN_DDP_IMPL must be 'manual' or 'ddp'")
        if batch_metrics_log_interval < 1:
            raise ValueError("batch_metrics_log_interval must be at least 1")
        self.cumulative_wall = 0
        logging.debug("* Initialize Trainer")

        # store all init arguments
        self.model = model

        _local_kwargs = {}
        for key in self.init_keys:
            setattr(self, key, locals()[key])
            _local_kwargs[key] = locals()[key]

        self.ema = None

        output = Output.get_output(dict(**_local_kwargs, **kwargs))
        self.output = output

        self.logfile = output.open_logfile("log", propagate=True)
        self.epoch_log = output.open_logfile("metrics_epoch.csv", propagate=False)
        self.init_epoch_log = output.open_logfile(
            "metrics_initialization.csv", propagate=False
        )
        self.batch_log = {
            TRAIN: output.open_logfile(
                f"metrics_batch_{ABBREV[TRAIN]}.csv", propagate=False
            ),
            VALIDATION: output.open_logfile(
                f"metrics_batch_{ABBREV[VALIDATION]}.csv", propagate=False
            ),
        }
        self._batch_log_header_written = {TRAIN: False, VALIDATION: False}

        # add filenames if not defined
        self.best_model_path = output.generate_file("best_model.pth")
        self.last_model_path = output.generate_file("last_model.pth")
        self.trainer_save_path = output.generate_file("trainer.pth")
        self.config_path = self.output.generate_file("config.yaml")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # seed python random as well so checkpointed runs can reproduce
            # all three RNG streams exactly
            random.seed(seed)

        self.dataset_rng = torch.Generator()
        if dataset_seed is not None:
            self.dataset_rng.manual_seed(dataset_seed)

        self.logger.info(f"Torch device: {self.device}")
        self.torch_device = torch.device(self.device)

        # sort out all the other parameters
        # for samplers, optimizer and scheduler
        self.kwargs = deepcopy(kwargs)
        self.optimizer_kwargs = deepcopy(optimizer_kwargs)
        self.lr_scheduler_kwargs = deepcopy(lr_scheduler_kwargs)
        self.early_stopping_kwargs = deepcopy(early_stopping_kwargs)

        # ``None`` preserves the historical DDP defaults.  ``find_unused``
        # additionally preserves the existing environment-variable switch
        # when its config key is absent; an explicit config value wins.
        self._ddp_options = {
            "broadcast_buffers": (
                True
                if self.ddp_broadcast_buffers is None
                else self.ddp_broadcast_buffers
            ),
            "static_graph": (
                False if self.ddp_static_graph is None else self.ddp_static_graph
            ),
            "gradient_as_bucket_view": (
                False
                if self.ddp_gradient_as_bucket_view is None
                else self.ddp_gradient_as_bucket_view
            ),
            "find_unused_parameters": (
                os.environ.get("SPINGNN_DDP_FIND_UNUSED", "1") != "0"
                if self.ddp_find_unused_parameters is None
                else self.ddp_find_unused_parameters
            ),
        }

        # initialize training states
        self.best_metrics = float("inf")
        self.best_epoch = 0
        self.iepoch = -1 if self.report_init_validation else 0
        loss_kwargs = dict(
            coeffs=self.loss_coeffs,
            ldetach_method=self.ldetach_method,
            lauto_loss=self.lauto_loss,
        )
        if self.use_e2IP_loss:
            loss_kwargs.update(
                evidential_keys=[self.evidential_key],
                reg=self.loss_reg,
                epsilon=self.loss_epsilon,
                spin_force_scale=self.e2ip_spin_force_scale,
                likelihood_scale=self.e2ip_likelihood_scale,
                likelihood_dtype=self.e2ip_likelihood_dtype,
            )
        self.loss, _ = instantiate(
            builder=e2IPLoss if self.use_e2IP_loss else Loss,
            prefix="loss",
            positional_args=loss_kwargs,
            all_args=self.kwargs,
        )

        validation_loss_coeffs = (
            self.validation_loss_coeffs
            if self.validation_loss_coeffs is not None
            else self.loss_coeffs
        )
        validation_loss_kwargs = dict(
            coeffs=validation_loss_coeffs,
            ldetach_method=self.ldetach_method,
            lauto_loss=False,
        )
        if self.validation_use_e2IP_loss:
            validation_loss_kwargs.update(
                evidential_keys=[self.evidential_key],
                reg=self.loss_reg,
                epsilon=self.loss_epsilon,
                spin_force_scale=self.e2ip_spin_force_scale,
                likelihood_scale=self.e2ip_likelihood_scale,
                likelihood_dtype=self.e2ip_likelihood_dtype,
            )
        self.validation_loss, _ = instantiate(
            builder=e2IPLoss if self.validation_use_e2IP_loss else Loss,
            prefix="loss",
            positional_args=validation_loss_kwargs,
            all_args=self.kwargs,
        )

        if self._ddp_enabled:
            if self.lauto_loss:
                raise RuntimeError(
                    "automatic weighted loss (lauto_loss) is not supported in "
                    "distributed training; its parameters live outside the "
                    "synchronized model. Disable it for distributed runs."
                )
            if self.ldetach_method:
                raise RuntimeError(
                    "ldetach_method is not supported in distributed training; "
                    "its detached normalization has no globally additive contract"
                )
            if str(self.optimizer_name).lower() == "sam":
                raise RuntimeError(
                    "SAM is not supported in distributed training; its second "
                    "forward/backward uses the rank-local mean loss, which has "
                    "no globally additive contract. Disable it for "
                    "distributed runs."
                )
            self.loss.assert_distributed_compatible()
            self.validation_loss.assert_distributed_compatible()
            self.loss_stat = DistributedLossStat(
                self.loss, device=self._ddp_comm_device()
            )
            self.validation_loss_stat = DistributedLossStat(
                self.validation_loss, device=self._ddp_comm_device()
            )
        else:
            self.loss_stat = LossStat(self.loss)
            self.validation_loss_stat = LossStat(self.validation_loss)

        # what do we train on?
        self.train_on_keys = self.loss.keys
        if train_on_keys is not None:
            assert set(train_on_keys) == set(self.train_on_keys)
        self._remove_from_model_input = set(self.train_on_keys)
        if (
            len(
                self._remove_from_model_input.intersection(
                    AtomicDataDict.ALL_ENERGY_KEYS
                )
            )
            > 0
        ):
            # if we are training on _any_ of the energy quantities (energy, force, partials, stress, etc.)
            # then none of them should be fed into the model
            self._remove_from_model_input = self._remove_from_model_input.union(
                AtomicDataDict.ALL_ENERGY_KEYS
            )
        if kwargs.get("_override_allow_truth_label_inputs", False):
            # needed for unit testing models
            self._remove_from_model_input = set()

        # load all callbacks
        self._init_callbacks = [load_callable(callback) for callback in init_callbacks]
        self._start_of_epoch_callbacks = [
            load_callable(callback) for callback in start_of_epoch_callbacks
        ]
        self._end_of_epoch_callbacks = [
            load_callable(callback) for callback in end_of_epoch_callbacks
        ]
        self._end_of_batch_callbacks = [
            load_callable(callback) for callback in end_of_batch_callbacks
        ]
        self._end_of_train_callbacks = [
            load_callable(callback) for callback in end_of_train_callbacks
        ]
        self._final_callbacks = [
            load_callable(callback) for callback in final_callbacks
        ]

        self.init()

    def init_objects(self):
        # initialize optimizer
        opt_opt_args = dict(lr=self.learning_rate)
        if self.optimizer_kwargs is not None:
            opt_opt_args.update(self.optimizer_kwargs)
        self._optimizer_group_structure = None
        if self.optimizer_param_groups is not None:
            if self.train_spin_ladder or self.train_ztype_eng_ladder:
                # ladders unfreeze parameters later; groups hold only the
                # parameters trainable when the optimizer is built
                raise ValueError(
                    "optimizer_param_groups cannot be combined with "
                    "train_spin_ladder or train_ztype_eng_ladder"
                )
            # names are resolved on the unwrapped model (no DDP ``module.``)
            (
                optimizer_params,
                self._optimizer_group_structure,
            ) = build_optimizer_param_groups(
                self._unwrapped_model(), self.optimizer_param_groups
            )
            if hasattr(self.loss, "awl"):
                optimizer_params.append({"params": self.loss.awl.parameters()})
        else:
            optimizer_params = self.model.parameters()
            if hasattr(self.loss, "awl"):
                optimizer_params = [
                    {"params": self.model.parameters()},
                    {"params": self.loss.awl.parameters()},
                ]
        self.optim, self._resolved_optimizer_kwargs = instantiate_from_cls_name(
            module=[torch.optim, nequip.utils.optim],
            class_name=self.optimizer_name,
            prefix="optimizer",
            positional_args=dict(params=optimizer_params),
            optional_args=opt_opt_args,
            all_args=self.kwargs,
        )
        del opt_opt_args
        if self.optimizer_param_groups is not None:
            check_param_group_options(self.optim, self.optimizer_param_groups)
            for group in self._optimizer_group_structure:
                self.logger.info(
                    f"Optimizer param group {group['name']}: "
                    f"{len(group['params'])} tensors"
                )
        self._early_stopping_lr_group_index = resolve_param_group_index(
            self.optim, self.early_stopping_lr_group
        )
        self._using_SAM = isinstance(self.optim, nequip.utils.optim.SAM)

        self.max_gradient_norm = (
            float(self.max_gradient_norm)
            if self.max_gradient_norm is not None
            else float("inf")
        )

        # initialize scheduler
        assert (
            self.lr_scheduler_name
            in ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "none"]
        ) or (
            (len(self.end_of_epoch_callbacks) + len(self.end_of_batch_callbacks)) > 0
        ), f"{self.lr_scheduler_name} cannot be used unless callback functions are defined"

        if self.train_spin_ladder:
            print("Use Ladder training strategy")
            if self.train_spin_ladder_list is None:
                self.train_spin_ladder_list = [0]
                if self.lhb:
                    if self.train_spin_ladder_only_frozen:
                        self.train_spin_ladder_list.extend([1])
                    else:
                        self.train_spin_ladder_list.extend([1, 2])
                if self.lnosoc_layer:
                    if self.train_spin_ladder_only_frozen:
                        self.train_spin_ladder_list.extend([3])
                    else:
                        self.train_spin_ladder_list.extend([3, 4])
                if self.lsoc_layer:
                    if self.train_spin_ladder_only_frozen:
                        self.train_spin_ladder_list.extend([5])
                    else:
                        self.train_spin_ladder_list.extend([5, 6])

            print(
                len(self.train_spin_ladder_list),
                "ladders include: ",
                self.train_spin_ladder_list,
            )

            self.spin_ladder_index = 0
            self.model = tune_model_coeff_nosoc(
                self.model, state=self.train_spin_ladder_list[self.spin_ladder_index]
            )
            self.init_early_stopping(prefix="early_stopping_ladder")
            self.init_lr_scheduler(prefix="lr_scheduler_ladder")
        elif self.train_ztype_eng_ladder:
            print("Use Ladder training strategy with E_type + E_NN")
            if self.train_ztype_eng_ladder_full:
                self.train_ztype_eng_ladder_list = [0, 1, 2]
            else:
                self.train_ztype_eng_ladder_list = [0, 1]
            print(
                len(self.train_ztype_eng_ladder_list),
                "ladders include: ",
                self.train_ztype_eng_ladder_list,
            )

            self.ladder_index = 0
            self.model = tune_model_ztype(
                self.model, state=self.train_ztype_eng_ladder_list[self.ladder_index]
            )
            self.init_early_stopping(prefix="early_stopping_ladder")
            self.init_lr_scheduler(prefix="lr_scheduler_ladder")
        else:
            self.init_early_stopping()
            self.init_lr_scheduler()

        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.ema_decay,
                use_num_updates=self.ema_use_num_updates,
            )

        if hasattr(self.model, "irreps_out"):
            for key in self.train_on_keys:
                if key not in self.model.irreps_out:
                    raise RuntimeError(
                        f"Loss function include fields {key} that are not predicted by the model {self.model.irreps_out}"
                    )

    def init_lr_scheduler(self, prefix="lr_scheduler"):
        self.lr_sched = None
        self._resolved_lr_scheduler_kwargs = {}
        if self.lr_scheduler_name != "none":
            (
                self.lr_sched,
                self._resolved_lr_scheduler_kwargs,
            ) = instantiate_from_cls_name(
                module=torch.optim.lr_scheduler,
                class_name=self.lr_scheduler_name,
                prefix=prefix,
                positional_args=dict(
                    optimizer=self.optim.base_optimizer
                    if self._using_SAM  # see SAM https://github.com/davda54/sam README.md
                    else self.optim
                ),
                optional_args=self.lr_scheduler_kwargs,
                all_args=self.kwargs,
            )

    def init_early_stopping(self, prefix="early_stopping"):
        # initialize early stopping conditions
        key_mapping, kwargs = instantiate(
            EarlyStopping,
            prefix=prefix,
            optional_args=self.early_stopping_kwargs,
            all_args=self.kwargs,
            return_args_only=True,
        )
        n_args = 0
        # per-group learning-rate columns (optimizer_param_groups) are epoch
        # quantities like LR, not validation metrics
        lr_columns = set(self._group_lr_columns())
        for key, item in kwargs.items():
            # prepend VALIDATION string if k is not with
            if isinstance(item, dict):
                new_dict = {}
                for k, v in item.items():
                    if (
                        k.lower().startswith(VALIDATION)
                        or k.lower().startswith(TRAIN)
                        or k.lower() in ["lr", "wall", "cumulative_wall"]
                        or k in lr_columns
                    ):
                        new_dict[k] = item[k]
                    else:
                        new_dict[f"{VALIDATION}_{k}"] = item[k]
                kwargs[key] = new_dict
                n_args += len(new_dict)
        self.early_stopping_conds = EarlyStopping(**kwargs) if n_args > 0 else None

    @property
    def init_keys(self):
        return [
            key
            for key in list(inspect.signature(Trainer.__init__).parameters.keys())
            if key not in (["self", "kwargs", "model"] + Trainer.object_keys)
        ]

    @property
    def params(self):
        return self.as_dict(state_dict=False, training_progress=False, kwargs=False)

    def update_kwargs(self, config):
        self.kwargs.update(
            {key: value for key, value in config.items() if key not in self.init_keys}
        )

    @property
    def logger(self):
        return logging.getLogger(self.logfile)

    @property
    def epoch_logger(self):
        return logging.getLogger(self.epoch_log)

    @property
    def init_epoch_logger(self):
        return logging.getLogger(self.init_epoch_log)

    def as_dict(
        self,
        state_dict: bool = False,
        training_progress: bool = False,
        kwargs: bool = True,
        recovery_generation: Optional[str] = None,
    ):
        """convert instance to a dictionary
        Args:

        state_dict (bool): if True, the state_dicts of the optimizer, lr scheduler, and EMA will be included
        """

        dictionary = {}

        for key in self.init_keys:
            dictionary[key] = getattr(self, key, None)

        if kwargs:
            dictionary.update(getattr(self, "kwargs", {}))

        if state_dict:
            dictionary["state_dict"] = {}
            for key in Trainer.object_keys:
                item = getattr(self, key, None)
                if item is not None:
                    dictionary["state_dict"][key] = item.state_dict()
            if hasattr(self.loss, "awl"):
                dictionary["state_dict"]["loss_awl"] = self.loss.awl.state_dict()
            if getattr(self, "_optimizer_group_structure", None) is not None:
                # named group layout, checked against the rebuilt optimizer on
                # resume (torch only checks group sizes)
                dictionary["state_dict"]["optim_param_groups"] = deepcopy(
                    self._optimizer_group_structure
                )
            if not self._ddp_enabled:
                # single-process checkpoints embed the RNG state directly;
                # distributed checkpoints keep it in per-rank files next to
                # the canonical file (see _save_distributed_rank_state)
                dictionary["state_dict"]["rng_state"] = torch.get_rng_state()
                dictionary["state_dict"][
                    "dataset_rng_state"
                ] = self.dataset_rng.get_state()
                if torch.cuda.is_available():
                    dictionary["state_dict"][
                        "cuda_rng_state"
                    ] = torch.cuda.get_rng_state(device=self.torch_device)
            dictionary["state_dict"]["cumulative_wall"] = self.cumulative_wall

        if training_progress:
            dictionary["progress"] = {}
            for key in ["iepoch", "best_epoch"]:
                dictionary["progress"][key] = self.__dict__.get(key, -1)
            dictionary["progress"]["best_metrics"] = self.__dict__.get(
                "best_metrics", float("inf")
            )
            dictionary["progress"]["stop_arg"] = self.__dict__.get("stop_arg", None)

            # TODO: these might not both be available, str defined, but no weights
            dictionary["progress"]["best_model_path"] = self.best_model_path
            dictionary["progress"]["last_model_path"] = self.last_model_path
            dictionary["progress"]["trainer_save_path"] = self.trainer_save_path
            if hasattr(self, "config_save_path"):
                dictionary["progress"]["config_save_path"] = self.config_save_path

        if training_progress and self._ddp_enabled:
            # canonical recovery metadata linking this file to the per-rank
            # RNG files of one checkpoint generation
            dictionary["recovery"] = {
                "schema_version": _RECOVERY_SCHEMA_VERSION,
                "generation": recovery_generation,
                "world_size": self._ddp_world_size,
                "completed_epoch": self.iepoch,
                "rank_files": {
                    rank: _RANK_STATE_FILE_PATTERN.format(rank=rank)
                    for rank in range(self._ddp_world_size)
                },
                "sampler": {
                    "shuffle": self.shuffle,
                    "seed": int(self.dataset_seed or 0),
                    "train_frames": len(self.dataset_train)
                    if getattr(self, "dataset_train", None) is not None
                    else None,
                    "val_frames": len(self.dataset_val)
                    if getattr(self, "dataset_val", None) is not None
                    else None,
                    "train_padding": self._ddp_train_padding,
                },
            }

        return dictionary

    def save_config(self, blocking: bool = True) -> None:
        save_file(
            item=self.as_dict(state_dict=False, training_progress=False),
            supported_formats=dict(yaml=["yaml"]),
            filename=self.config_path,
            enforced_format=None,
            blocking=blocking,
        )

    def save(self, filename: Optional[str] = None, format=None, blocking: bool = True):
        """save the file as filename

        Args:

        filename (str): name of the file
        format (str): format of the file. yaml and json format will not save the weights.
        """

        if filename is None:
            filename = self.trainer_save_path

        logger = self.logger

        state_dict = (
            True
            if format == "torch"
            or filename.endswith(".pth")
            or filename.endswith(".pt")
            else False
        )

        recovery_generation = None
        if self._ddp_enabled and state_dict:
            # canonical + rank-local checkpoint contract: every rank commits
            # its RNG file first; rank 0 then writes the canonical trainer
            # file (and model weights) as the manifest of the generation
            recovery_generation = self._save_distributed_rank_state()
            if self._ddp_rank != 0:
                return filename

        filename = save_file(
            item=self.as_dict(
                state_dict=state_dict,
                training_progress=True,
                recovery_generation=recovery_generation,
            ),
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
            blocking=blocking,
        )
        logger.debug(f"Saved trainer to {filename}")

        self.save_model(self.last_model_path, blocking=blocking)
        logger.debug(f"Saved last model to to {self.last_model_path}")

        return filename

    def _ddp_canonical_dir(self) -> Path:
        """Directory that owns the canonical checkpoint (rank 0's workdir)."""

        return (
            Path(self.kwargs.get("ddp_rank0_root", self.output.root))
            / self.output.run_name
        )

    def _save_distributed_rank_state(self) -> str:
        """Commit this rank's RNG file and return the checkpoint generation.

        Protocol: rank 0 draws a generation id and broadcasts it; every rank
        writes its rank-local file (temp file + rename) next to the canonical
        checkpoint; a barrier then makes all rank files durable before rank 0
        writes the canonical trainer file (the manifest, written last) in
        ``save``.
        """

        generation = [str(uuid.uuid4()) if self._ddp_rank == 0 else None]
        _ddp_broadcast_object(generation)
        rank_state = {
            "schema_version": _RECOVERY_SCHEMA_VERSION,
            "generation": generation[0],
            "rank": self._ddp_rank,
            "world_size": self._ddp_world_size,
            "device": str(self.torch_device),
            "rng_state": torch.get_rng_state(),
            "dataset_rng_state": self.dataset_rng.get_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device=self.torch_device)
                if torch.cuda.is_available() and self.torch_device.type == "cuda"
                else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        rank_path = self._ddp_canonical_dir() / _RANK_STATE_FILE_PATTERN.format(
            rank=self._ddp_rank
        )
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = rank_path.with_name(rank_path.name + ".tmp")
        with open(temporary, "wb") as file:
            torch.save(rank_state, file)
        os.replace(temporary, rank_path)
        dist.barrier()
        return generation[0]

    def _restore_rank_rng_state(self, rank_state) -> None:
        """Restore this rank's own RNG state from its rank-local file.

        Runs last in the restart sequence: after bare-weight loading, DDP
        reconstruction, and optimizer/EMA/scheduler state restore, so any
        RNG consumed during construction is overwritten by the checkpointed
        state. Exact recovery is scoped to epoch boundaries with
        ``dataloader_num_workers=0``; persistent dataloader workers and
        stochastic in-worker transforms have their own RNG state that this
        contract does not capture.
        """

        torch.set_rng_state(rank_state["rng_state"])
        self.dataset_rng.set_state(rank_state["dataset_rng_state"])
        cuda_state = rank_state.get("cuda_rng_state")
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "the checkpoint contains CUDA RNG state but CUDA is "
                    "unavailable in this process; restart on the original "
                    "device type"
                )
            torch.cuda.set_rng_state(cuda_state, device=self.torch_device)
        np.random.set_state(rank_state["numpy_rng_state"])
        random.setstate(rank_state["python_rng_state"])

    @classmethod
    def from_file(
        cls, filename: str, format: Optional[str] = None, append: Optional[bool] = None
    ):
        """load a model from file

        Args:

        filename (str): name of the file
        append (bool): if True, append the old model files and append the same logfile
        """

        dictionary = load_file(
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
            map_location="cpu",
        )
        return cls.from_dict(dictionary, append)

    @classmethod
    def from_dict(cls, dictionary, append: Optional[bool] = None):
        """load model from dictionary

        Args:

        dictionary (dict):
        append (bool): if True, append the old model files and append the same logfile
        """

        dictionary = deepcopy(dictionary)
        check_code_version(dictionary)

        # update the restart and append option
        if append is not None:
            dictionary["append"] = append

        recovery = dictionary.get("recovery", None)
        rank_state = None
        if recovery is not None:
            # canonical + rank-local checkpoint contract (distributed):
            # validate before any construction so every rank fails together
            rank_state = cls._validate_distributed_checkpoint(dictionary, recovery)

        model = None
        iepoch = -1
        if "model" in dictionary:
            model = dictionary.pop("model")
        elif "progress" in dictionary:
            progress = dictionary["progress"]

            # load the model from file
            iepoch = progress["iepoch"]
            if isfile(progress["last_model_path"]):
                load_path = Path(progress["last_model_path"])
                iepoch = progress["iepoch"]
            elif recovery is not None:
                raise RuntimeError(
                    "canonical model weights are missing at "
                    f"{progress['last_model_path']}; the checkpoint "
                    "generation is incomplete — restart from the previous "
                    "complete checkpoint"
                )
            else:
                raise AttributeError("model weights & bias are not saved")

            model, _ = Trainer.load_model_from_training_session(
                traindir=load_path.parent,
                model_name=load_path.name,
                config_dictionary=dictionary,
            )
            logging.debug(f"Reload the model from {load_path}")

        state_dict = dictionary.pop("state_dict", None)

        trainer = cls(model=model, **dictionary)

        if state_dict is not None and trainer.model is not None:
            logging.debug("Reload optimizer and scheduler states")
            if "optim" in state_dict:
                mismatch = describe_structure_mismatch(
                    state_dict.get("optim_param_groups", None),
                    getattr(trainer, "_optimizer_group_structure", None),
                )
                if mismatch is not None:
                    raise RuntimeError(
                        "optimizer param groups of the checkpoint do not match the "
                        "rebuilt optimizer (optimizer_param_groups or the model's "
                        f"parameter names changed): {mismatch}"
                    )
            for key in Trainer.object_keys:
                item = getattr(trainer, key, None)
                if item is not None:
                    item_state_dict = state_dict[key]
                    if key == "optim" and hasattr(trainer.loss, "awl"):
                        current_param_groups = item.state_dict()["param_groups"]
                        saved_param_groups = item_state_dict["param_groups"]
                        if len(saved_param_groups) == len(current_param_groups) - 1:
                            # Checkpoints created before AWL was optimized contain only
                            # the model parameter group. Add an empty AWL group so that
                            # those runs remain resumable.
                            item_state_dict = deepcopy(item_state_dict)
                            item_state_dict["param_groups"].append(
                                current_param_groups[-1]
                            )
                    item.load_state_dict(item_state_dict)
            if hasattr(trainer.loss, "awl") and "loss_awl" in state_dict:
                trainer.loss.awl.load_state_dict(state_dict["loss_awl"])
            trainer._initialized = True
            trainer.cumulative_wall = state_dict["cumulative_wall"]

            if rank_state is not None:
                # restore this rank's own RNG last: after bare-weight
                # loading, DDP reconstruction, and optimizer/EMA/scheduler
                # state restore (see _restore_rank_rng_state)
                trainer._restore_rank_rng_state(rank_state)
            else:
                torch.set_rng_state(state_dict["rng_state"])
                trainer.dataset_rng.set_state(state_dict["dataset_rng_state"])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(state_dict["cuda_rng_state"])

        reset_best_metrics = dictionary.get("reset_best_metrics", False)
        if "progress" in dictionary:
            stop_arg = progress.pop("stop_arg", None)
            if reset_best_metrics:
                trainer.best_metrics = float("inf")
                trainer.best_epoch = 0
                logging.info("Reset best_metrics/best_epoch for finetuning run")
            else:
                trainer.best_metrics = progress["best_metrics"]
                trainer.best_epoch = progress["best_epoch"]
        else:
            trainer.best_metrics = float("inf")
            trainer.best_epoch = 0
            stop_arg = None
        trainer.iepoch = iepoch

        # final sanity check
        if trainer.stop_cond:
            raise RuntimeError(
                f"The previous run has properly stopped with {stop_arg}."
                "Please either increase the max_epoch or change early stop criteria"
            )

        return trainer

    @classmethod
    def _validate_distributed_checkpoint(cls, dictionary, recovery):
        """Validate a canonical + rank-local checkpoint before loading it.

        Every rank checks the canonical metadata and its own rank file, then
        an all-reduce makes all ranks fail together so that no rank enters
        DDP construction while another rank has already rejected the
        checkpoint. Returns the validated rank-local RNG state.
        """

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "this checkpoint was written by distributed training "
                "(recovery metadata is present) but the current process is "
                "not distributed; restart it under torchrun with the "
                "original world size"
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        error = None
        rank_state = None
        schema_version = recovery.get("schema_version")
        if schema_version != _RECOVERY_SCHEMA_VERSION:
            error = (
                f"unsupported checkpoint recovery schema version "
                f"{schema_version} (this code supports "
                f"{_RECOVERY_SCHEMA_VERSION}); restart with the code version "
                "that wrote the checkpoint"
            )
        elif recovery.get("world_size") != world_size:
            error = (
                f"checkpoint was written with world size "
                f"{recovery.get('world_size')} but the current run has world "
                f"size {world_size}; restarting with a different world size "
                "is not supported — relaunch with the original world size"
            )
        else:
            progress = dictionary.get("progress", {})
            canonical_dir = Path(str(progress.get("trainer_save_path", "."))).parent
            rank_path = canonical_dir / _RANK_STATE_FILE_PATTERN.format(rank=rank)
            last_model_path = progress.get("last_model_path")
            if not rank_path.is_file():
                error = (
                    f"missing rank-local RNG state file {rank_path} for rank "
                    f"{rank}; the checkpoint generation is incomplete — "
                    "restart from the previous complete checkpoint"
                )
            elif last_model_path is None or not isfile(last_model_path):
                error = (
                    f"canonical model weights are missing at "
                    f"{last_model_path}; the checkpoint generation is "
                    "incomplete — restart from the previous complete "
                    "checkpoint"
                )
            else:
                rank_state = torch.load(
                    rank_path, map_location="cpu", weights_only=False
                )
                if rank_state.get("generation") != recovery.get("generation"):
                    error = (
                        "checkpoint generation mismatch: the canonical "
                        f"trainer file has generation "
                        f"{recovery.get('generation')} but the rank {rank} "
                        f"file {rank_path} has generation "
                        f"{rank_state.get('generation')}; mixed checkpoint "
                        "generations — restart from one consistent generation"
                    )
                elif (
                    rank_state.get("rank") != rank
                    or rank_state.get("world_size") != world_size
                ):
                    error = (
                        f"rank-local RNG state file {rank_path} describes "
                        f"rank {rank_state.get('rank')} of world size "
                        f"{rank_state.get('world_size')} but this is rank "
                        f"{rank} of world size {world_size}; do not move "
                        "rank files between ranks"
                    )
                state_dict = dictionary.get("state_dict") or {}
                if error is None and "optim" not in state_dict:
                    error = (
                        "checkpoint is missing optimizer state; it was "
                        "probably saved before the first completed epoch — "
                        "restart from a checkpoint written after at least "
                        "one completed epoch"
                    )
                if error is not None:
                    rank_state = None
        comm_device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        failures = torch.tensor(
            [0 if error is None else 1], dtype=torch.long, device=comm_device
        )
        dist.all_reduce(failures, op=dist.ReduceOp.SUM)
        if failures.item() > 0:
            raise RuntimeError(
                error
                if error is not None
                else (
                    "distributed checkpoint validation failed on another "
                    "rank; check that rank's log for the root cause"
                )
            )
        return rank_state

    @staticmethod
    def load_model_from_training_session(
        traindir,
        model_name="best_model.pth",
        device="cpu",
        config_dictionary: Optional[dict] = None,
    ) -> Tuple[torch.nn.Module, Config]:
        traindir = str(traindir)
        model_name = str(model_name)

        if config_dictionary is not None:
            config = Config.from_dict(config_dictionary)
        else:
            config = Config.from_file(traindir + "/config.yaml")

        model = model_from_config(
            config=config,
            initialize=False,
        )
        if model is not None:  # TODO: why would it be?
            from nequip.utils._global_options import (
                cast_model_dtype,
                model_dtype_from_config,
            )

            # TODO: this is not exactly equivalent to building with
            # this set as default dtype... does it matter?
            # ``model_dtype`` (default: ``default_dtype``); modules marked with
            # ``_spingnn_keep_dtype`` keep their own dtypes
            cast_model_dtype(
                model,
                model_dtype_from_config(config),
                device=torch.device(device),
            )
            model_state_dict = torch.load(
                traindir + "/" + model_name,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(model_state_dict)

        return model, config

    def init(self):
        """initialize optimizer"""
        if self.model is None:
            return

        self.model.to(self.torch_device)
        if hasattr(self.loss, "awl"):
            reference_param = next(
                (p for p in self.model.parameters() if p.is_floating_point()), None
            )
            if reference_param is None:
                self.loss.awl.to(device=self.torch_device)
            else:
                self.loss.awl.to(
                    device=self.torch_device,
                    dtype=reference_param.dtype,
                )

        self.num_weights = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Number of weights: {self.num_weights}")
        self.logger.info(
            f"Number of trainable weights: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        self.rescale_layers = []
        outer_layer = self.model
        while hasattr(outer_layer, "unscale"):
            self.rescale_layers.append(outer_layer)
            outer_layer = getattr(outer_layer, "model", None)

        if self._ddp_enabled and self._ddp_impl == "manual":
            self._ddp_validate_initial_model_state()

        # Native DDP is the default distributed implementation. The torch
        # 2.11/cu128 DDP-construction crash on RTX 5090/4090 is avoided by
        # the communicator warmup below (probe jobs 2290284/2290286 crash
        # without it vs 2290293/2290294 pass with it; native DDP trains and
        # learns on both GPU types in jobs 2290420/2290421/2290422).
        # SPINGNN_DDP_IMPL=manual remains as an explicit diagnostic path.
        if (
            self._ddp_enabled
            and self._ddp_impl == "ddp"
            and not isinstance(self.model, DistributedDataParallel)
        ):
            # Guard native DDP construction with the communicator warmup (see
            # warmup_process_group_for_ddp); idempotent, so the train.py CLI
            # warmup and this direct-construction path share one broadcast.
            warmup_process_group_for_ddp()
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.torch_device.index]
                if self.torch_device.type == "cuda"
                else None,
                output_device=self.torch_device.index
                if self.torch_device.type == "cuda"
                else None,
                **self._ddp_options,
            )

        self.init_objects()

        self._initialized = True
        self.cumulative_wall = 0

    def _ddp_validate_initial_model_state(self):
        """Fail closed if rank-local deterministic initialization diverges."""

        checksum = torch.zeros((), device=self.torch_device, dtype=torch.float32)
        for parameter in self.model.parameters():
            checksum.add_(parameter.detach().float().sum())
        summed = checksum.clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        average = summed / self._ddp_world_size
        if not torch.allclose(checksum, average, rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError(
                "distributed ranks produced different initial model parameters; "
                "use identical explicit seeds on every rank"
            )

    def _ddp_reduce_gradients(self):
        """Average gradients across ranks for the manual data-parallel path."""

        if not self._ddp_enabled or self._ddp_impl != "manual":
            return
        parameters = list(self.model.parameters())
        # All ranks must reduce the same parameter slots, even when local
        # supervision leaves different heads unused. Preserve None only for
        # globally unused parameters so optimizers do not decay those heads.
        used = torch.tensor(
            [parameter.grad is not None for parameter in parameters],
            dtype=torch.int32,
            device=self.torch_device,
        )
        dist.all_reduce(used, op=dist.ReduceOp.SUM)
        for parameter, count in zip(parameters, used.cpu().tolist()):
            if count == 0:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self._ddp_world_size)

    def init_metrics(self):
        if self.metrics_components is None:
            self.metrics_components = []
            for key, func in self.loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.lower().startswith("perspecies"),
                }
                self.metrics_components.append((key, "mae", params))
                self.metrics_components.append((key, "rmse", params))

        self.metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(
                components=self.metrics_components,
                distributed=self._ddp_enabled,
                species_count=(
                    getattr(
                        getattr(
                            getattr(self, "dataset_train", None),
                            "type_mapper",
                            None,
                        ),
                        "num_types",
                        None,
                    )
                    if self._ddp_enabled
                    else None
                ),
            ),
            all_args=self.kwargs,
        )

        if not (
            self.metrics_key.lower().startswith(VALIDATION)
            or self.metrics_key.lower().startswith(TRAIN)
        ):
            raise RuntimeError(
                f"metrics_key should start with either {VALIDATION} or {TRAIN}"
            )
        if self.report_init_validation and self.metrics_key.lower().startswith(TRAIN):
            raise RuntimeError(
                f"metrics_key may not start with {TRAIN} when report_init_validation=True"
            )

    def _ddp_comm_device(self):
        """Device for small statistics collectives (gloo requires CPU)."""

        if dist.get_backend() == "nccl":
            return self.torch_device
        return torch.device("cpu")

    def _ddp_sync_buffers(self):
        """Broadcast rank-0 buffers once before unwrapped validation forwards.

        The unwrapped validation path runs no DDP forward, so the default
        buffer broadcast never happens; keep buffer state rank-consistent
        explicitly before the loop.
        """

        if not self._ddp_options["broadcast_buffers"]:
            return
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = model.module
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

    def train(self):
        """Training"""
        from nequip.utils._global_options import global_options_context

        with global_options_context(self.kwargs):
            return self._train()

    def _train(self):
        if getattr(self, "dl_train", None) is None:
            raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        if not self._initialized:
            self.init()

        if self._ddp_enabled and self._ddp_rank != 0:
            # Rank zero owns the output files; all ranks still execute the
            # training loop and participate in DDP collectives.
            self.logger.setLevel(logging.WARNING)

        for callback in self._init_callbacks:
            callback(self)

        self.init_log()
        self.wall = perf_counter()
        self.previous_cumulative_wall = self.cumulative_wall

        with atomic_write_group():
            if self.iepoch == -1:
                self.save()
            if self.iepoch in [-1, 0]:
                self.save_config()

        self.init_metrics()

        while not self.stop_cond:

            self.epoch_step()
            self.end_of_epoch_save()

        for callback in self._final_callbacks:
            callback(self)

        self.final_log()

        self.save()
        finish_all_writes()

    def batch_step(self, data, validation=False):
        # no need to have gradients from old steps taking up memory
        self.optim.zero_grad(set_to_none=True)

        if validation:
            self.model.eval()
        else:
            self.model.train()

        # Do any target rescaling
        data = data.to(self.torch_device)
        data = AtomicData.to_AtomicDataDict(data)

        data_unscaled = data
        for layer in self.rescale_layers:
            # This means that self.model is RescaleOutputs
            # this will normalize the targets
            # in validation (eval mode), it does nothing
            # in train mode, if normalizes the targets
            data_unscaled = layer.unscale(data_unscaled)

        # Run model
        # We make a shallow copy of the input dict in case the model modifies it
        input_data = {
            k: v
            for k, v in data_unscaled.items()
            if k not in self._remove_from_model_input
        }
        # Validation runs through the unwrapped module when distributed: no
        # per-batch collectives, autograd stays enabled (forces are
        # derivatives), buffers were synced once before the loop.
        forward_model = self.model
        if (
            validation
            and self._ddp_enabled
            and isinstance(forward_model, DistributedDataParallel)
        ):
            forward_model = forward_model.module
        with training_autocast_context(self.kwargs, self.torch_device.type):
            out = forward_model(input_data)
        del input_data
        # If we're in evaluation mode (i.e. validation), then
        # data_unscaled's target prop is unnormalized, and out's has been rescaled to be in the same units
        # If we're in training, data_unscaled's target prop has been normalized, and out's hasn't been touched, so they're both in normalized units
        # Note that either way all normalization was handled internally by RescaleOutput

        if not validation:
            # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            self.optim.zero_grad(set_to_none=True)
            # Actually do an optimization step, since we're training:
            # bfloat16 has the same exponent range as float32, so this path
            # intentionally does not use GradScaler.
            with training_autocast_context(self.kwargs, self.torch_device.type):
                if self._ddp_enabled:
                    loss, loss_numerators, loss_counts = distributed_objective(
                        self.loss,
                        pred=out,
                        ref=data_unscaled,
                        world_size=self._ddp_world_size,
                    )
                else:
                    loss, loss_contrib = self.loss(pred=out, ref=data_unscaled)
            loss.backward()
            self._ddp_reduce_gradients()
            if self._using_SAM:
                self.optim.first_step(zero_grad=True)
                # note that for the second step, which uses the modified
                # weights from SAM, we do not store the loss for metrics
                # And we also throw out the output dict
                # See README.md of https://github.com/davda54/sam also
                # Again note that we do not save this loss to pass to
                # loss metrics / logging, since this loss is largely
                # meaningless and the above loss (before the first
                # step) is the one that tracks with the error metrics
                # on the model before the first optimization weight
                # update step. (Ref. personal discussions.)
                with training_autocast_context(self.kwargs, self.torch_device.type):
                    sam_loss = self.loss(
                        pred=self.model(data_unscaled.copy()),
                        ref=data_unscaled,
                    )[0]
                sam_loss.backward()
                self._ddp_reduce_gradients()

            # See https://stackoverflow.com/a/56069467
            # Has to happen after .backward() so there are grads to clip
            if self.max_gradient_norm < float("inf"):
                if self._using_SAM:
                    raise NotImplementedError(
                        "Gradient clipping is not currently supported when SAM is enabled"
                    )
                parameters = self.model.parameters()
                if hasattr(self.loss, "awl"):
                    parameters = (
                        parameter
                        for group in self.optim.param_groups
                        for parameter in group["params"]
                    )
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    self.max_gradient_norm,
                )

            if self._using_SAM:
                # grads get zero'd at start of next loop
                self.optim.second_step(zero_grad=False)
            else:
                self.optim.step()

            if self.use_ema:
                self.ema.update()

            if self.lr_scheduler_name == "CosineAnnealingWarmRestarts":
                self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)
        # end train block

        with torch.no_grad():
            if validation:
                scaled_out = out
                _data_unscaled = data
                for layer in self.rescale_layers:
                    # loss function always needs to be in normalized unit
                    scaled_out = layer.unscale(scaled_out, force_process=True)
                    _data_unscaled = layer.unscale(_data_unscaled, force_process=True)
                with training_autocast_context(self.kwargs, self.torch_device.type):
                    if self._ddp_enabled:
                        (
                            loss_numerators,
                            loss_counts,
                        ) = self.validation_loss.sum_and_count(
                            pred=scaled_out, ref=_data_unscaled
                        )
                    else:
                        loss, loss_contrib = self.validation_loss(
                            pred=scaled_out, ref=_data_unscaled
                        )
                current_loss_stat = self.validation_loss_stat
            else:
                # If we are in training mode, we need to bring the prediction
                # into real units
                for layer in self.rescale_layers[::-1]:
                    out = layer.scale(out, force_process=True)
                current_loss_stat = self.loss_stat

            # save metrics stats
            materialize_batch = self._should_materialize_batch_log()
            if self._ddp_enabled:
                self.batch_losses = current_loss_stat.accumulate(
                    loss_numerators, loss_counts, materialize=materialize_batch
                )
            else:
                self.batch_losses = current_loss_stat(
                    loss, loss_contrib, materialize=materialize_batch
                )
            # in validation mode, data is in real units and the network scales
            # out to be in real units interally.
            # in training mode, data is still in real units, and we rescaled
            # out to be in real units above.
            self.batch_metrics = self.metrics(
                pred=out, ref=data, materialize=materialize_batch
            )
            if not materialize_batch:
                self.batch_metrics = None

    def _should_materialize_batch_log(self):
        """Whether this batch needs scalar metrics for the batch log."""

        interval = self.batch_metrics_log_interval
        return (
            interval == 1
            or (self.ibatch + 1) % interval == 0
            or (self.ibatch + 1) == self.n_batches
        )

    @property
    def stop_cond(self):
        """kill the training early"""

        if self.early_stopping_conds is not None and hasattr(self, "mae_dict"):
            early_stop, early_stop_args, debug_args = self.early_stopping_conds(
                self.mae_dict
            )
            if debug_args is not None:
                self.logger.debug(debug_args)
            if early_stop:
                print(json.dumps(self.mae_dict, sort_keys=False, indent=4))
                if self.train_spin_ladder:
                    self.spin_ladder_index = self.spin_ladder_index + 1

                    if self.spin_ladder_index == len(self.train_spin_ladder_list):
                        self.stop_arg = early_stop_args
                        return True  # Last state end
                    else:
                        state = self.train_spin_ladder_list[self.spin_ladder_index]
                        print("tune_model_coeff_nosoc state: {}".format(state))
                        self.model = tune_model_coeff_nosoc(self.model, state=state)
                        print(
                            "Reset early stopping and learning rate at epoch: ",
                            self.iepoch,
                        )
                        if (
                            self.spin_ladder_index
                            == len(self.train_spin_ladder_list) - 1
                        ):
                            # In last ladder, we use original early stopping and lr scheduler
                            self.init_early_stopping(prefix="early_stopping")
                            self.init_lr_scheduler(prefix="lr_scheduler")
                        else:
                            self.init_early_stopping(prefix="early_stopping_ladder")
                            self.init_lr_scheduler(prefix="lr_scheduler_ladder")
                        if state in [2, 4, 6]:
                            self.optim.param_groups[0][
                                "lr"
                            ] = self.learning_rate_fine_tune
                        else:
                            self.optim.param_groups[0]["lr"] = self.learning_rate
                        return False
                elif self.train_ztype_eng_ladder:
                    self.ladder_index = self.ladder_index + 1

                    if self.ladder_index == len(self.train_ztype_eng_ladder_list):
                        self.stop_arg = early_stop_args
                        return True  # Last state end
                    else:
                        state = self.train_ztype_eng_ladder_list[self.ladder_index]
                        logging.info("tune_model_ztype state: {}".format(state))
                        self.model = tune_model_ztype(self.model, state=state)
                        print(
                            "Reset early stopping and learning rate at epoch: ",
                            self.iepoch,
                        )
                        if state == 2:
                            # In last ladder, we use original early stopping and lr scheduler
                            self.init_early_stopping(prefix="early_stopping")
                            self.init_lr_scheduler(prefix="lr_scheduler")
                        else:
                            self.init_early_stopping(prefix="early_stopping_ladder")
                            self.init_lr_scheduler(prefix="lr_scheduler_ladder")
                        if state in [
                            2,
                        ]:
                            self.optim.param_groups[0][
                                "lr"
                            ] = self.learning_rate_fine_tune
                        else:
                            self.optim.param_groups[0]["lr"] = self.learning_rate
                        return False
                else:
                    self.stop_arg = early_stop_args
                    return True

        if self.iepoch >= self.max_epochs:
            self.stop_arg = "max epochs"
            return True

        return False

    def reset_metrics(self, validation: bool = False):
        loss_stat = self.validation_loss_stat if validation else self.loss_stat
        loss_stat.reset()
        loss_stat.to(self.torch_device)
        self.metrics.reset()
        self.metrics.to(self.torch_device)

    def epoch_step(self):

        if self._ddp_enabled:
            if self._ddp_train_sampler is not None:
                self._ddp_train_sampler.set_epoch(max(self.iepoch, 0))
            if self._ddp_val_sampler is not None:
                self._ddp_val_sampler.set_epoch(max(self.iepoch, 0))

        if self.iepoch >= 0:
            # start of a training epoch; after a restart this is the first
            # call, made once all checkpointed state has been restored
            for callback in self._start_of_epoch_callbacks:
                callback(self)

        dataloaders = {TRAIN: self.dl_train, VALIDATION: self.dl_val}
        categories = [TRAIN, VALIDATION] if self.iepoch >= 0 else [VALIDATION]
        dataloaders = [
            dataloaders[c] for c in categories
        ]  # get the right dataloaders for the catagories we actually run
        self.metrics_dict = {}
        self.loss_dict = {}

        for category, dataset in zip(categories, dataloaders):
            if category == VALIDATION and self.use_ema:
                cm = self.ema.average_parameters()
            else:
                cm = contextlib.nullcontext()

            with cm:
                is_validation = category == VALIDATION
                if is_validation and self._ddp_enabled:
                    self._ddp_sync_buffers()
                current_loss_stat = (
                    self.validation_loss_stat if is_validation else self.loss_stat
                )
                self.reset_metrics(validation=is_validation)
                self.n_batches = len(dataset)
                for self.ibatch, batch in enumerate(dataset):
                    self.batch_step(
                        data=batch,
                        validation=is_validation,
                    )
                    self.end_of_batch_log(batch_type=category)
                    for callback in self._end_of_batch_callbacks:
                        callback(self)
                if self._ddp_enabled:
                    # one final aggregation of additive sufficient statistics
                    self.metrics_dict[category] = self.metrics.global_result()
                    self.loss_dict[category] = current_loss_stat.global_result()
                else:
                    self.metrics_dict[category] = self.metrics.current_result()
                    self.loss_dict[category] = current_loss_stat.current_result()

                if category == TRAIN:
                    for callback in self._end_of_train_callbacks:
                        callback(self)

        self.iepoch += 1

        self.end_of_epoch_log()

        # if the iepoch for the past epoch was -1, it will now be 0
        # for -1 (report_init_validation: True) we aren't training, so it's wrong
        # to step the LR scheduler even if it will have no effect with this particular
        # scheduler at the beginning of training.
        if self.iepoch > 0 and self.lr_scheduler_name == "ReduceLROnPlateau":
            self.lr_sched.step(metrics=self.mae_dict[self.metrics_key])

        for callback in self._end_of_epoch_callbacks:
            callback(self)

    def end_of_batch_log(self, batch_type: str):
        """
        store all the loss/mae of each batch
        """

        if self.batch_metrics is None:
            return
        if self._ddp_enabled and self._ddp_rank != 0:
            return

        mat_str = f"{self.iepoch+1:5d}, {self.ibatch+1:5d}"
        log_str = f"  {self.iepoch+1:5d} {self.ibatch+1:5d}"

        header = "epoch, batch"
        log_header = "# Epoch batch"

        # print and store loss value
        for name, value in self.batch_losses.items():
            mat_str += f", {value:16.5g}"
            header += f", {name}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12.12}"

        # append details from metrics
        metrics, skip_keys = self.metrics.flatten_metrics(
            metrics=self.batch_metrics,
            type_names=self.dataset_train.type_mapper.type_names,
        )
        for key, value in metrics.items():

            mat_str += f", {value:16.5g}"
            header += f", {key}"
            if key not in skip_keys:
                log_str += f" {value:12.3g}"
                log_header += f" {key:>12.12}"

        batch_logger = logging.getLogger(self.batch_log[batch_type])

        if self.ibatch == 0:
            self.logger.info("")
            self.logger.info(f"{batch_type}")
            self.logger.info(log_header)

        init_step = -1 if self.report_init_validation else 0
        write_header = (self.iepoch == init_step and batch_type == VALIDATION) or (
            self.iepoch == 0 and batch_type == TRAIN
        )
        if write_header and not self._batch_log_header_written[batch_type]:
            batch_logger.info(header)
            self._batch_log_header_written[batch_type] = True

        batch_logger.info(mat_str)
        if (self.ibatch + 1) % self.log_batch_freq == 0 or (
            self.ibatch + 1
        ) == self.n_batches:
            self.logger.info(log_str)

    def end_of_epoch_save(self):
        """
        save model and trainer details
        """
        with atomic_write_group():
            current_metrics = self.mae_dict[self.metrics_key]
            if current_metrics < self.best_metrics:
                self.best_metrics = current_metrics
                self.best_epoch = self.iepoch

                self.save_ema_model(self.best_model_path, blocking=False)

                self.logger.info(
                    f"! Best model {self.best_epoch:8d} {self.best_metrics:8.3f}"
                )

            if (self.iepoch + 1) % self.log_epoch_freq == 0:
                self.save(blocking=False)

            if (
                self.save_checkpoint_freq > 0
                and (self.iepoch + 1) % self.save_checkpoint_freq == 0
            ):
                ckpt_path = self.output.generate_file(f"ckpt{self.iepoch+1}.pth")
                self.save_model(ckpt_path, blocking=False)

            if (
                self.save_ema_checkpoint_freq > 0
                and (self.iepoch + 1) % self.save_ema_checkpoint_freq == 0
            ):
                ckpt_path = self.output.generate_file(f"ckpt_ema_{self.iepoch+1}.pth")
                self.save_ema_model(ckpt_path, blocking=False)

    def save_ema_model(self, path, blocking: bool = True):

        if self.use_ema:
            # If using EMA, store the EMA validation model
            # that gave us the good val metrics that made the model "best"
            # in the first place
            cm = self.ema.average_parameters()
        else:
            # otherwise, do nothing
            cm = contextlib.nullcontext()

        with cm:
            self.save_model(path, blocking=blocking)

    def _group_lr_columns(self):
        """``{"LR_<group name>": lr}`` for ``optimizer_param_groups``, else ``{}``."""
        if getattr(self, "_optimizer_group_structure", None) is None:
            return {}
        return {
            f"LR_{layout['name']}": group["lr"]
            for layout, group in zip(
                self._optimizer_group_structure, self.optim.param_groups
            )
        }

    def _unwrapped_model(self):
        """The model without a ``DistributedDataParallel`` wrapper."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def save_model(self, path, blocking: bool = True):
        if self._ddp_enabled and self._ddp_rank != 0:
            return
        model = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        with atomic_write(path, blocking=blocking, binary=True) as write_to:
            torch.save(model.state_dict(), write_to)

    def init_log(self):
        if self.iepoch > 0:
            self.logger.info("! Restarting training ...")
        else:
            self.logger.info("! Starting training ...")

    def final_log(self):

        self.logger.info(f"! Stop training: {self.stop_arg}")
        wall = perf_counter() - self.wall
        self.cumulative_wall = wall + self.previous_cumulative_wall
        self.logger.info(f"Wall time: {wall}")
        self.logger.info(f"Cumulative wall time: {self.cumulative_wall}")

    def end_of_epoch_log(self):
        """
        log validation details at the end of each epoch
        """

        if self._ddp_enabled:
            dist.barrier()
            # Every rank computes the reduced metrics below so that stopping
            # and scheduler decisions remain synchronized. Output roots are
            # rank-local; rank zero is the retained publication output.

        lr = self.optim.param_groups[self._early_stopping_lr_group_index]["lr"]
        wall = perf_counter() - self.wall
        self.cumulative_wall = wall + self.previous_cumulative_wall
        self.mae_dict = dict(
            LR=lr,
            epoch=self.iepoch,
            wall=wall,
            cumulative_wall=self.cumulative_wall,
        )

        header = "epoch, wall, LR"

        categories = [TRAIN, VALIDATION] if self.iepoch > 0 else [VALIDATION]
        log_str = {}

        strings = ["Epoch", "wal", "LR"]
        mat_str = f"{self.iepoch:10d}, {wall:8.3f}, {lr:8.3g}"
        # with optimizer_param_groups, one extra LR_<name> column per group
        # (metrics CSV and mae_dict); the LR column above is unchanged
        for column, group_lr in self._group_lr_columns().items():
            self.mae_dict[column] = group_lr
            header += f", {column}"
            mat_str += f", {group_lr:8.3g}"
        loss_keys = []
        metric_keys = []
        category_results = {}

        for category in categories:

            met, skip_keys = self.metrics.flatten_metrics(
                metrics=self.metrics_dict[category],
                type_names=self.dataset_train.type_mapper.type_names,
            )
            skip_keys = set(skip_keys)
            category_results[category] = {
                "losses": self.loss_dict[category],
                "metrics": met,
                "skip_keys": skip_keys,
            }

            # append details from loss
            for key, value in self.loss_dict[category].items():
                mat_str += f", {value:16.5g}"
                header += f",{category}_{key}"
                self.mae_dict[f"{category}_{key}"] = value
                if key not in loss_keys:
                    loss_keys.append(key)

            # append details from metrics
            for key, value in met.items():
                mat_str += f", {value:12.3g}"
                header += f",{category}_{key}"
                self.mae_dict[f"{category}_{key}"] = value
                if key not in skip_keys and key not in metric_keys:
                    metric_keys.append(key)

        log_header = "# " + " ".join([f"{s:>8s}" for s in strings])
        for key in loss_keys:
            log_header += f" {key:>12.12}"
        for key in metric_keys:
            log_header += f" {key:>12.12}"

        for category in categories:
            log_str[category] = f"{self.iepoch:10d} {wall:8.3f} {lr:8.3g}"
            losses = category_results[category]["losses"]
            metrics = category_results[category]["metrics"]
            skip_keys = category_results[category]["skip_keys"]

            for key in loss_keys:
                if key in losses:
                    log_str[category] += f" {losses[key]:12.3g}"
                else:
                    log_str[category] += f" {'':>12}"

            for key in metric_keys:
                if key in metrics and key not in skip_keys:
                    log_str[category] += f" {metrics[key]:12.3g}"
                else:
                    log_str[category] += f" {'':>12}"

        if self.iepoch == 0:
            self.init_epoch_logger.info(header)
            self.init_epoch_logger.info(mat_str)
        elif self.iepoch == 1:
            self.epoch_logger.info(header)

        if self.iepoch > 0:
            self.epoch_logger.info(mat_str)

        if self.iepoch > 0:
            self.logger.info("\n\n  Train      " + log_header)
            self.logger.info("! Train      " + log_str[TRAIN])
            self.logger.info("! Validation " + log_str[VALIDATION])
        else:
            self.logger.info("\n\n  Initialization     " + log_header)
            self.logger.info("! Initial Validation " + log_str[VALIDATION])

        if hasattr(self.loss, "awl"):
            awl_params = self.loss.awl.params.detach().cpu().tolist()
            self.logger.info(f"! AWL parameters {awl_params}")

        wall = perf_counter() - self.wall
        self.logger.info(f"Wall time: {wall}")

    def __del__(self):

        if not self._initialized:
            return

        logger = self.logger
        for hdl in logger.handlers:
            hdl.flush()
            hdl.close()
        logger.handlers = []

        for i in range(len(logger.handlers)):
            logger.handlers.pop()

    def set_dataset(
        self,
        dataset: AtomicDataset,
        validation_dataset: Optional[AtomicDataset] = None,
    ) -> None:
        """Set the dataset(s) used by this trainer.

        Training and validation datasets will be sampled from
        them in accordance with the trainer's parameters.

        If only one dataset is provided, the train and validation
        datasets will both be sampled from it. Otherwise, if
        `validation_dataset` is provided, it will be used.
        """

        if self.train_idcs is None or self.val_idcs is None:
            if validation_dataset is None:
                # Sample both from `dataset`:
                total_n = len(dataset)
                if (self.n_train + self.n_val) > total_n:
                    raise ValueError(
                        "too little data for training and validation. please reduce n_train and n_val"
                    )

                if self.train_val_split == "random":
                    idcs = torch.randperm(total_n, generator=self.dataset_rng)
                elif self.train_val_split == "sequential":
                    idcs = torch.arange(total_n)
                else:
                    raise NotImplementedError(
                        f"splitting mode {self.train_val_split} not implemented"
                    )

                self.train_idcs = idcs[: self.n_train]
                self.val_idcs = idcs[self.n_train : self.n_train + self.n_val]
            else:
                if self.n_train > len(dataset):
                    raise ValueError("Not enough data in dataset for requested n_train")
                if self.n_val > len(validation_dataset):
                    raise ValueError(
                        "Not enough data in validation dataset for requested n_val"
                    )
                if self.train_val_split == "random":
                    self.train_idcs = torch.randperm(
                        len(dataset), generator=self.dataset_rng
                    )[: self.n_train]
                    self.val_idcs = torch.randperm(
                        len(validation_dataset), generator=self.dataset_rng
                    )[: self.n_val]
                elif self.train_val_split == "sequential":
                    self.train_idcs = torch.arange(self.n_train)
                    self.val_idcs = torch.arange(self.n_val)
                else:
                    raise NotImplementedError(
                        f"splitting mode {self.train_val_split} not implemented"
                    )

        if validation_dataset is None:
            validation_dataset = dataset

        # torch_geometric datasets inherantly support subsets using `index_select`
        self.dataset_train = dataset.index_select(self.train_idcs)
        self.dataset_val = validation_dataset.index_select(self.val_idcs)

        if self._ddp_enabled:
            self._ddp_train_sampler = DistributedSampler(
                self.dataset_train,
                num_replicas=self._ddp_world_size,
                rank=self._ddp_rank,
                shuffle=self.shuffle,
                seed=int(self.dataset_seed or 0),
                drop_last=False,
            )
            # training pads to equal-length shards; record the padding so the
            # consumed-sample statistics are interpretable
            self._ddp_train_padding = len(
                self._ddp_train_sampler
            ) * self._ddp_world_size - len(self.dataset_train)
            # validation uses deterministic nonpadding slices instead of a
            # padding sampler; ranks may have unequal (or zero) batch counts
            self._ddp_val_sampler = None
            self._ddp_val_indices = range(
                self._ddp_rank, len(self.dataset_val), self._ddp_world_size
            )
            if self._ddp_rank == 0:
                self.logger.info(
                    f"Distributed sampling: {len(self.dataset_train)} train frames "
                    f"({self._ddp_train_padding} padding), "
                    f"{len(self.dataset_val)} validation frames over "
                    f"{self._ddp_world_size} ranks"
                )

        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        dl_kwargs = dict(
            exclude_keys=self.exclude_keys,
            num_workers=self.dataloader_num_workers,
            # keep stuff around in memory
            persistent_workers=(
                self.dataloader_num_workers > 0 and self.max_epochs > 1
            ),
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=(self.torch_device != torch.device("cpu")),
            # avoid getting stuck
            timeout=(10 if self.dataloader_num_workers > 0 else 0),
            # use the right randomness
            generator=self.dataset_rng,
        )
        self.dl_train = DataLoader(
            dataset=self.dataset_train,
            shuffle=self.shuffle and not self._ddp_enabled,
            sampler=self._ddp_train_sampler,
            batch_size=self.batch_size,
            **dl_kwargs,
        )
        # validation, on the other hand, shouldn't shuffle
        # we still pass the generator just to be safe
        self.dl_val = DataLoader(
            dataset=self.dataset_val,
            sampler=self._ddp_val_indices if self._ddp_enabled else None,
            batch_size=self.validation_batch_size,
            **dl_kwargs,
        )
