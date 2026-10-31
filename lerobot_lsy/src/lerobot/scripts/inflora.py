#!/usr/bin/env python
"""InfLoRA continual learning for the DiT flow policy, without adapter routing.

Replaces CLARE's expansion and autoencoder routing by InfLoRA (Liang & Li, CVPR 2024,
https://github.com/liangyanshuo/InfLoRA). For every target nn.Linear y = W x + b and task t:

    y = W x + b + B_t A_t x        A_t: (rank, d_in), fixed;  B_t: (d_out, rank), zero init, trained

1. Before training, the input covariance C = mean(x x^T) of each target layer is gathered on the new task's data.
2. A_t is the top `rank` left singular vectors of C projected away from the input subspace of the old tasks
   (kept by DualGPM), divided by sqrt(3) as in the official code. Training B_t then only changes W along
   directions the old tasks barely use, so it interferes little with them.
3. After training, B_t A_t is merged into W, so the model has the same size after every task and needs no task
   id or routing at deployment. DualGPM is then updated with the covariance of the new task.

Each stage (one task) is one call of this script, like clare.py. The stage saves `inflora_state.pt` in its output
dir (the low-rank factors of every learned task and the DualGPM bases, a few MB); the next stage loads it through
--inflora_state_path. No full model checkpoint is written: the merged policy is the pretrained policy plus the
saved factors.

With --orthogonal=false, A_t comes from the unprojected covariance of the new task (sequential LoRA with the same
data-driven A, no protection of old tasks): the ablation that isolates the effect of the projection.

The evaluation after each stage logs `success_<task>:<rate>` like clare.py, so clare_metrics.py reads the logs.
"""

import copy
import logging
import math
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_identifier
from lerobot.utils.utils import format_big_number, get_safe_torch_device, has_method, init_logging
from lerobot.utils.wandb_utils import WandBLogger

STATE_FILE = "inflora_state.pt"


@dataclass
class InfLoRATrainPipelineConfig(TrainPipelineConfig):
    # state saved by the previous stage (its STATE_FILE or the folder holding it); None for the first task
    inflora_state_path: Path | None = None
    # regex, matched in full against the policy's module names; every matching nn.Linear gets a LoRA branch.
    # Default: the layer CLARE adapts. Add the decoder MLPs with
    # ".*velocity_net\.(cond_proj|decoder\.layers\.\d+\.linear[12])"
    target_modules: str = r".*velocity_net\.cond_proj"
    rank: int = 32
    # DualGPM energy threshold, raised linearly from lamb (first task) towards lame (official: 0.95 and 1.0)
    lamb: float = 0.95
    lame: float = 1.0
    total_tasks: int = 10
    # project A_t away from the old tasks' input subspace (InfLoRA); false = plain sequential LoRA (ablation)
    orthogonal: bool = True
    # batches used to estimate the input covariance of each target layer (CLARE's detection uses 200 x 32)
    feature_steps: int = 200
    feature_batch_size: int = 32
    max_episodes_rendered: int = 4


class InfLoRALinear(nn.Module):
    """nn.Linear plus one trainable LoRA branch B A x, A fixed. Can also accumulate the input covariance."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.rank = rank
        weight = base.weight
        self.register_buffer("lora_A", torch.zeros(rank, base.in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=weight.device, dtype=weight.dtype))
        self.register_buffer("cov", torch.zeros(base.in_features, base.in_features, device=weight.device,
                                                dtype=torch.float64), persistent=False)
        self.n_cov = 0
        self.collect = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.collect:
            flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float64)
            n = flat.shape[0]
            self.cov.mul_(self.n_cov / (self.n_cov + n)).add_(flat.T @ flat / (self.n_cov + n))
            self.n_cov += n
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B)

    def reset_cov(self) -> None:
        self.cov.zero_()
        self.n_cov = 0

    @torch.no_grad()
    def merge(self) -> None:
        """W += B A, then reset B so that the forward pass is unchanged."""
        self.base.weight.add_(self.lora_B @ self.lora_A)
        self.lora_B.zero_()


def wrap_target_layers(policy: nn.Module, pattern: str, rank: int) -> dict[str, InfLoRALinear]:
    regex = re.compile(pattern)
    names = [name for name, module in policy.named_modules()
             if regex.fullmatch(name) and isinstance(module, nn.Linear)]
    if not names:
        raise ValueError(f"no nn.Linear module matches target_modules={pattern!r}")
    layers = {}
    for name in names:
        layer = InfLoRALinear(policy.get_submodule(name), rank)
        policy.set_submodule(name, layer)
        layers[name] = layer
    return layers


@torch.no_grad()
def collect_covariance(cfg: InfLoRATrainPipelineConfig, policy: PreTrainedPolicy, layers: dict, dataset,
                       device) -> dict[str, np.ndarray]:
    """Mean of x x^T over the inputs of each target layer, on feature_steps batches of the task's data."""
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.feature_batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    data_iter = cycle(loader)
    policy.eval()
    for layer in layers.values():
        layer.reset_cov()
        layer.collect = True
    for _ in range(cfg.feature_steps):
        batch = next(data_iter)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        policy.forward(batch)
    covs = {}
    for name, layer in layers.items():
        layer.collect = False
        covs[name] = layer.cov.cpu().numpy()
        layer.reset_cov()
    return covs


def init_lora_A(cov: np.ndarray, basis: np.ndarray | None, project_type: str | None, rank: int) -> np.ndarray:
    """Top `rank` directions of the new task's input covariance, outside the old tasks' subspace (official code:
    cur_matrix - M M^T cur_matrix for 'remove', M M^T cur_matrix for 'retain'), scaled by 1/sqrt(3)."""
    cur = cov
    if basis is not None:
        proj = basis @ basis.T
        cur = cur - proj @ cur if project_type == "remove" else proj @ cur
    u, _, _ = np.linalg.svd(cur, full_matrices=False)
    return u[:, :rank].T / math.sqrt(3)


def update_dual_gpm(activation: np.ndarray, basis: np.ndarray | None, project_type: str | None,
                    threshold: float) -> tuple[np.ndarray, str, int]:
    """One layer of InfLoRA's update_DualGPM, ported as is (the 'activation' is the input covariance, as in the
    official code). 'remove': basis spans the old tasks' input space; 'retain': basis spans its complement.
    Returns the new basis, its type and the number of directions added (0 = layer skipped)."""
    if basis is None:
        u, s, _ = np.linalg.svd(activation, full_matrices=False)
        ratio = s**2 / (s**2).sum()
        r = int(np.sum(np.cumsum(ratio) < threshold))
        # The official code labels U[:, :r] 'retain' when r >= d/2 (its own assert then fails); store it as
        # 'remove' and let the check below swap it for its complement.
        basis, project_type, added = u[:, : max(r, 1)], "remove", r
    elif project_type == "remove":
        _, s1, _ = np.linalg.svd(activation, full_matrices=False)
        total = (s1**2).sum()
        act_hat = activation - basis @ (basis.T @ activation)
        u, s, _ = np.linalg.svd(act_hat, full_matrices=False)
        ratio = s**2 / total
        accumulated = (total - (s**2).sum()) / total
        r = 0
        for value in ratio:
            if accumulated >= threshold:
                break
            accumulated += value
            r += 1
        added = r
        if r:
            basis = np.hstack((basis, u[:, :r]))[:, : activation.shape[0]]
    else:
        _, s1, _ = np.linalg.svd(activation, full_matrices=False)
        total = (s1**2).sum()
        act_hat = basis @ (basis.T @ activation)
        u, s, _ = np.linalg.svd(act_hat, full_matrices=False)
        ratio = s**2 / total
        accumulated = (s**2).sum() / total
        r = 0
        for value in ratio:
            if accumulated < 1 - threshold:
                break
            accumulated -= value
            r += 1
        added = r
        if r:
            act_feature = basis - u[:, :r] @ (u[:, :r].T @ basis)
            ui, _, _ = np.linalg.svd(act_feature)
            basis = ui[:, : basis.shape[1] - r]
    # keep whichever of the subspace and its complement is smaller
    if project_type == "remove" and basis.shape[1] > basis.shape[0] / 2:
        u, _, _ = np.linalg.svd(basis)
        basis, project_type = u[:, basis.shape[1]:], "retain"
    return basis, project_type, added


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm, error_if_nonfinite=False)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def evaluate(cfg: InfLoRATrainPipelineConfig, policy: PreTrainedPolicy, eval_envs: dict, dataset, step: int,
             device, wandb_logger) -> None:
    """Roll out every task learned so far and log success_<task> like clare.py (clare_metrics.py parses it)."""
    step_id = get_step_identifier(step, cfg.steps)
    eval_metrics = {
        "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
        "pc_success": AverageMeter("success", ":.1f"),
        "eval_s": AverageMeter("eval_s", ":.3f"),
    }
    eval_infos = {}
    policy.eval()
    for task, eval_env_cfg in eval_envs.items():
        logging.info(f"Eval task {task}")
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            eval_infos[task] = eval_policy_with_env_init(
                eval_env_cfg,
                cfg.eval.batch_size,
                False,
                policy,
                cfg.eval.n_episodes,
                videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                max_episodes_rendered=cfg.max_episodes_rendered,
                start_seed=cfg.seed,
            )
        eval_metrics[f"avg_sum_reward_{task}"] = AverageMeter(f"∑rwrd_{task}", ":.3f")
        eval_metrics[f"pc_success_{task}"] = AverageMeter(f"success_{task}", ":.1f")
        eval_metrics[f"eval_s_{task}"] = AverageMeter(f"eval_s_{task}", ":.3f")

    eval_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics,
                                  initial_step=step)
    sums = {"avg_sum_reward": 0.0, "pc_success": 0.0, "eval_s": 0.0}
    for task, eval_info in eval_infos.items():
        for key in sums:
            value = eval_info["aggregated"].pop(key)
            sums[key] += value
            eval_tracker.__setattr__(f"{key}_{task}", value)
    eval_tracker.avg_sum_reward = sums["avg_sum_reward"] / len(eval_infos)
    eval_tracker.pc_success = sums["pc_success"] / len(eval_infos)
    eval_tracker.eval_s = sums["eval_s"]
    logging.info(eval_tracker)
    if wandb_logger:
        eval_info = eval_infos[list(eval_infos)[-1]]
        wandb_logger.log_dict({**eval_tracker.to_dict(), **eval_info}, step, mode="eval")
        wandb_logger.log_video(eval_info["video_paths"][-1], step, mode="eval")


@parser.wrap()
def train(cfg: InfLoRATrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project else None
    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    eval_envs = {}
    if cfg.env is not None:
        for task in [task.strip() for task in cfg.env.task.split(",") if task.strip()]:
            env_cfg = copy.deepcopy(cfg.env)
            env_cfg.task = task
            eval_envs[task] = env_cfg
        if not eval_envs:
            raise ValueError("No valid tasks found in env.task")

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    for p in policy.parameters():
        p.requires_grad = False
    layers = wrap_target_layers(policy, cfg.target_modules, cfg.rank)
    logging.info(f"InfLoRA layers: {list(layers)}")

    # Rebuild the merged weights of the tasks learned so far: W = W_pretrained + sum_t B_t A_t
    state = {"num_learned_task": 0, "factors": {name: [] for name in layers}, "basis": {}, "project_type": {}}
    if cfg.inflora_state_path is not None:
        path = Path(cfg.inflora_state_path)
        state = torch.load(path / STATE_FILE if path.is_dir() else path, map_location="cpu", weights_only=False)
        if set(state["factors"]) != set(layers):
            raise ValueError(f"state layers {sorted(state['factors'])} differ from target layers {sorted(layers)}")
        with torch.no_grad():
            for name, layer in layers.items():
                for lora_A, lora_B in state["factors"][name]:
                    layer.base.weight.add_(lora_B.to(layer.base.weight) @ lora_A.to(layer.base.weight))
    task_id = state["num_learned_task"]
    logging.info(f"Learning task {task_id} (tasks merged so far: {task_id})")

    logging.info("Initialising A from the input covariance of the new task")
    covs = collect_covariance(cfg, policy, layers, dataset, device)
    for name, layer in layers.items():
        basis = state["basis"].get(name) if cfg.orthogonal else None
        lora_A = init_lora_A(covs[name], basis, state["project_type"].get(name), cfg.rank)
        layer.lora_A.copy_(torch.from_numpy(lora_A).to(layer.lora_A))
        if basis is not None:
            # share of the new task's input energy inside the old tasks' subspace (projection 'remove' basis)
            proj = basis @ basis.T
            if state["project_type"][name] == "retain":
                proj = np.eye(proj.shape[0]) - proj
            old_share = float(np.trace(proj @ covs[name]) / np.trace(covs[name]))
            logging.info(f"{name}: old basis {basis.shape[1]}/{basis.shape[0]} ({state['project_type'][name]}), "
                         f"new task input energy in old subspace {old_share:.4f}")

    lora_params = [layer.lora_B for layer in layers.values()]
    optimizer = cfg.optimizer.build(lora_params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler else None
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    num_adapter_params = sum(p.numel() for p in lora_params)
    num_total_params = sum(p.numel() for name, p in policy.named_parameters() if "lora_B" not in name)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    # trainable during this stage; merged into the base weights afterwards, so the deployed size never grows
    logging.info(f"{num_adapter_params=} ({format_big_number(num_adapter_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(dataset.episode_data_index, drop_n_last_frames=cfg.policy.drop_n_last_frames,
                                      shuffle=True)
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics,
                                   initial_step=0)

    logging.info("Start offline training on a fixed dataset")
    # The base policy stays in eval mode (no dropout), as the frozen base layers in clare.py.
    policy.eval()
    for step in range(1, cfg.steps + 1):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )
        train_tracker.step()

        if cfg.log_freq > 0 and step % cfg.log_freq == 0:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if eval_envs and cfg.eval_freq > 0 and step % cfg.eval_freq == 0 and step < cfg.steps:
            logging.info(f"Eval policy at step {step}")
            evaluate(cfg, policy, eval_envs, dataset, step, device, wandb_logger)

    logging.info("Merging B A into the base weights")
    for name, layer in layers.items():
        state["factors"][name].append((layer.lora_A.detach().cpu().clone(), layer.lora_B.detach().cpu().clone()))
        layer.merge()

    # DualGPM update on the new task's inputs, after training as in the official code
    threshold = (cfg.lame - cfg.lamb) * task_id / cfg.total_tasks + cfg.lamb
    logging.info(f"Updating DualGPM, threshold {threshold:.4f}")
    covs = collect_covariance(cfg, policy, layers, dataset, device)
    for name in layers:
        basis, project_type, added = update_dual_gpm(covs[name], state["basis"].get(name),
                                                     state["project_type"].get(name), threshold)
        state["basis"][name], state["project_type"][name] = basis, project_type
        logging.info(f"{name}: +{added} directions, basis {basis.shape[1]}/{basis.shape[0]} type {project_type}")
    state["num_learned_task"] = task_id + 1

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, cfg.output_dir / STATE_FILE)
    logging.info(f"Saved {cfg.output_dir / STATE_FILE}")

    if eval_envs:
        logging.info(f"Eval policy at step {cfg.steps}")
        evaluate(cfg, policy, eval_envs, dataset, cfg.steps, device, wandb_logger)
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
