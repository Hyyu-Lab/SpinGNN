"""Static DDP contract checks for the training wrapper."""

from pathlib import Path


def test_trainer_initializes_distributed_sampler_slots_before_dataset_setup():
    source = (Path(__file__).parents[3] / "nequip/train/trainer.py").read_text()
    init_start = source.index("    def __init__(")
    init_end = source.index("    def init_objects", init_start)
    init = source[init_start:init_end]
    assert "self._ddp_train_sampler = None" in init
    assert "self._ddp_val_sampler = None" in init
    assert "DistributedSampler" in source
    assert "DistributedDataParallel" in source


def test_manual_ddp_uses_gradient_all_reduce_not_parameter_broadcast():
    source = (Path(__file__).parents[3] / "nequip/train/trainer.py").read_text()
    start = source.index("    def _ddp_validate_initial_model_state")
    end = source.index("    def init_metrics", start)
    manual = source[start:end]
    assert "dist.all_reduce" in manual
    assert "dist.broadcast" not in manual
    assert "dtype=torch.float32" in manual
    assert "_ddp_reduce_gradients" in source


def test_single_rank_torchrun_skips_distributed_init():
    """WORLD_SIZE=1 must not init NCCL (SIGSEGV on this stack, probe 2290340)."""
    root = Path(__file__).parents[3]
    train_source = (root / "nequip/scripts/train.py").read_text()
    init_start = train_source.index("def _initialize_distributed")
    init_end = train_source.index("def _prepare_distributed_config", init_start)
    init = train_source[init_start:init_end]
    single = init.index("WORLD_SIZE")
    pg = init.index("dist.init_process_group")
    assert single < pg, "the WORLD_SIZE==1 early return must precede init_process_group"


def test_ddp_construction_enables_find_unused_parameters_by_default():
    """allegro_soc smoke 2290758: unused params hard-fail the reducer."""
    root = Path(__file__).parents[3]
    trainer_source = (root / "nequip/train/trainer.py").read_text()
    assert '"find_unused_parameters": (' in trainer_source
    assert 'SPINGNN_DDP_FIND_UNUSED", "1"' in trainer_source
    assert '"broadcast_buffers"' in trainer_source
    assert '"static_graph"' in trainer_source
    assert '"gradient_as_bucket_view"' in trainer_source


def test_ddp_option_precedence_and_batch_interval_are_serialized_inputs():
    root = Path(__file__).parents[3]
    source = (root / "nequip/train/trainer.py").read_text()
    signature = source[
        source.index("    def __init__(") : source.index("    def init_objects")
    ]
    for name in (
        "ddp_broadcast_buffers",
        "ddp_static_graph",
        "ddp_gradient_as_bucket_view",
        "ddp_find_unused_parameters",
        "batch_metrics_log_interval",
    ):
        assert f"{name}:" in signature
    options = source[
        source.index("self._ddp_options") : source.index(
            "self.loss, _", source.index("self._ddp_options")
        )
    ]
    assert "self.ddp_find_unused_parameters is None" in options
    assert 'os.environ.get("SPINGNN_DDP_FIND_UNUSED", "1")' in options
    assert "self.ddp_find_unused_parameters" in options
    assert "batch_metrics_log_interval" in source


def test_ddp_warmup_broadcast_guards_native_ddp_construction():
    """Lock the NCCL warmup that unblocks torch 2.11/cu128 DDP construction."""
    root = Path(__file__).parents[3]

    train_source = (root / "nequip/scripts/train.py").read_text()
    init_start = train_source.index("def _initialize_distributed")
    init_end = train_source.index("def _prepare_distributed_config", init_start)
    init = train_source[init_start:init_end]
    assert init.index("dist.init_process_group") < init.index(
        "warmup_process_group_for_ddp()"
    ), "train.py must warm up the process group after init and before trainer construction"

    trainer_source = (root / "nequip/train/trainer.py").read_text()
    assert "def warmup_process_group_for_ddp" in trainer_source
    assert "dist.broadcast" in trainer_source
    guard_start = trainer_source.index("    def init(self):")
    guard_end = trainer_source.index(
        "    def _ddp_validate_initial_model_state", guard_start
    )
    guard = trainer_source[guard_start:guard_end]
    assert guard.index("warmup_process_group_for_ddp()") < guard.index(
        "DistributedDataParallel("
    ), "trainer.py must run the warmup before constructing DistributedDataParallel"
