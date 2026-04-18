"""Distributed integration tests for per-parameter mesh inference.

These tests spawn a 4-rank distributed job (single node, 4 GPUs) and
verify that Muon, NorMuon, and Dion2 correctly read each DTensor's
``device_mesh`` + ``placements`` without needing a global
``distributed_mesh`` argument.

Scenarios covered (all on a 2D mesh of shape (ep=2, efsdp=2)):

1. ``fsdp-only``     : plain FSDP on the efsdp sub-mesh
                      (2D matrix sharded on ``Shard(0)``)
2. ``ep-batch-shard``: MoE experts with ``[Shard(0)_ep, Shard(0)_efsdp]``
                      — both shards on the batch/experts dim.
                      No communication should be needed.
3. ``ep-matrix-shard``: MoE experts with ``[Shard(0)_ep, Shard(1)_efsdp]``
                      — EP on batch dim, efsdp on a matrix dim.
                      Only efsdp requires an all-to-all.
4. ``hsdp``          : non-experts with ``[Replicate()_ep, Shard(0)_efsdp]``
                      — HSDP-style.
5. ``mixed``         : all of the above in a single optimizer instance.

Each scenario runs the optimizer and asserts that parameters actually
changed (non-zero update magnitude) without raising.

Run with::

    cd /root/torchtitan-neo/dion
    torchrun --standalone --nproc_per_node=4 tests/test_distributed_meshes.py
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor


def _log(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    print(f"[rank{rank}] {msg}", flush=True)


def _make_param(
    shape: tuple[int, ...],
    mesh,
    placements,
    seed: int = 42,
) -> torch.nn.Parameter:
    """Create a DTensor parameter with deterministic content across ranks."""
    torch.manual_seed(seed)
    full = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    dt = distribute_tensor(full, mesh, placements)
    return torch.nn.Parameter(dt)


def _set_grad(param: torch.nn.Parameter, seed: int) -> None:
    torch.manual_seed(seed)
    if isinstance(param.data, DTensor):
        local = param.data.to_local()
        grad_local = torch.randn_like(local)
        param.grad = DTensor.from_local(
            grad_local,
            device_mesh=param.data.device_mesh,
            placements=param.data.placements,
        )
    else:
        param.grad = torch.randn_like(param)


def _assert_changed(param: torch.nn.Parameter, before: torch.Tensor, name: str) -> None:
    after = param.data.to_local() if isinstance(param.data, DTensor) else param.data
    delta = (after - before).abs().max().item()
    assert delta > 0, f"{name}: parameter did not change (max-abs-delta = {delta})"
    _log(f"{name}: param changed, max-abs-delta={delta:.4e}")


def _snapshot(param: torch.nn.Parameter) -> torch.Tensor:
    if isinstance(param.data, DTensor):
        return param.data.to_local().clone()
    return param.data.clone()


def run_scenario_fsdp_only(mesh_2d) -> None:
    from dion import Muon

    efsdp = mesh_2d["efsdp"]
    param = _make_param((64, 128), efsdp, (Shard(0),), seed=1)
    before = _snapshot(param)
    opt = Muon([param], lr=0.01)
    _set_grad(param, seed=101)
    opt.step()
    _assert_changed(param, before, "fsdp-only 2D [Shard(0)_efsdp]")


def run_scenario_ep_batch_shard(mesh_2d) -> None:
    """Experts with both shards on the batch dim (num_experts)."""
    from dion import Muon

    # Choose num_experts divisible by ep*efsdp=4.
    param = _make_param((8, 32, 64), mesh_2d, (Shard(0), Shard(0)), seed=2)
    before = _snapshot(param)
    opt = Muon([param], lr=0.01)
    _set_grad(param, seed=102)
    opt.step()
    _assert_changed(param, before, "ep-batch-shard 3D [Shard(0)_ep, Shard(0)_efsdp]")


def run_scenario_ep_matrix_shard(mesh_2d) -> None:
    """Experts with EP on batch dim and efsdp on a matrix dim."""
    from dion import Muon

    # num_experts=2 → ep shards it 1-per-rank on the ep axis; efsdp then
    # shards the d_in matrix dim across the efsdp axis.
    param = _make_param((2, 64, 128), mesh_2d, (Shard(0), Shard(1)), seed=3)
    before = _snapshot(param)
    opt = Muon([param], lr=0.01)
    _set_grad(param, seed=103)
    opt.step()
    _assert_changed(param, before, "ep-matrix-shard 3D [Shard(0)_ep, Shard(1)_efsdp]")


def run_scenario_hsdp(mesh_2d) -> None:
    """HSDP: Replicate on ep axis, Shard on efsdp."""
    from dion import Muon

    param = _make_param((64, 128), mesh_2d, (Replicate(), Shard(0)), seed=4)
    before = _snapshot(param)
    opt = Muon([param], lr=0.01)
    _set_grad(param, seed=104)
    opt.step()
    _assert_changed(param, before, "hsdp 2D [Replicate()_ep, Shard(0)_efsdp]")


def run_scenario_mixed(mesh_2d) -> None:
    """Non-expert (fsdp) + expert (batch-shard) + HSDP all in ONE Muon."""
    from dion import Muon

    efsdp = mesh_2d["efsdp"]
    non_expert = _make_param((64, 128), efsdp, (Shard(0),), seed=10)
    expert_batch = _make_param((8, 32, 64), mesh_2d, (Shard(0), Shard(0)), seed=11)
    expert_matrix = _make_param(
        (2, 64, 128), mesh_2d, (Shard(0), Shard(1)), seed=12
    )
    hsdp = _make_param((64, 128), mesh_2d, (Replicate(), Shard(0)), seed=13)

    params = [non_expert, expert_batch, expert_matrix, hsdp]
    names = ["non_expert-fsdp", "expert-batch", "expert-matrix", "hsdp"]
    befores = [_snapshot(p) for p in params]

    opt = Muon(params, lr=0.01)
    for i, p in enumerate(params):
        _set_grad(p, seed=200 + i)
    opt.step()

    for p, name, before in zip(params, names, befores):
        _assert_changed(p, before, f"mixed/{name}")


def run_scenario_mixed_optimizers(mesh_2d) -> None:
    """Verify Dion2 and NorMuon also handle mixed meshes."""
    from dion import Dion2, NorMuon

    efsdp = mesh_2d["efsdp"]
    for cls, name in [(Dion2, "Dion2"), (NorMuon, "NorMuon")]:
        non_expert = _make_param((64, 128), efsdp, (Shard(0),), seed=20)
        expert_batch = _make_param((8, 32, 64), mesh_2d, (Shard(0), Shard(0)), seed=21)
        params = [non_expert, expert_batch]
        befores = [_snapshot(p) for p in params]
        opt = cls(params, lr=0.01)
        for i, p in enumerate(params):
            _set_grad(p, seed=300 + i)
        opt.step()
        for p, pname, before in zip(params, ["non_expert", "expert-batch"], befores):
            _assert_changed(p, before, f"{name}/{pname}")


def main() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    assert world_size == 4, f"This test expects 4 ranks, got {world_size}"

    mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("ep", "efsdp"))

    scenarios = [
        ("fsdp-only", run_scenario_fsdp_only),
        ("ep-batch-shard", run_scenario_ep_batch_shard),
        ("ep-matrix-shard", run_scenario_ep_matrix_shard),
        ("hsdp", run_scenario_hsdp),
        ("mixed-in-one-muon", run_scenario_mixed),
        ("mixed-dion2-normuon", run_scenario_mixed_optimizers),
    ]

    failures: list[str] = []
    for name, fn in scenarios:
        _log(f"=== {name} ===")
        try:
            fn(mesh_2d)
            dist.barrier()
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"FAILED {name}: {e}\n{tb}")
            failures.append(name)
            try:
                dist.barrier()
            except Exception:
                pass

    if int(os.environ.get("RANK", "0")) == 0:
        if failures:
            print(f"\nFAILED scenarios: {failures}", flush=True)
        else:
            print("\nAll scenarios passed.", flush=True)

    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
