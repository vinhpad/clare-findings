#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import os

import gymnasium as gym

from lerobot.envs.configs import AlohaEnv, EnvConfig, LiberoEnv, HILEnvConfig, PushtEnv, XarmEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "hil":
        return HILEnvConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


class _FreezeWhenDone(gym.Wrapper):
    """FFI26 patch: once an episode has terminated or been truncated, return its last transition instead of
    stepping the simulator. Evaluation rollouts ignore everything after an env's first done step, so results
    are unchanged, but a batch no longer simulates every finished env until the slowest one is done."""

    def reset(self, **kwargs):
        if not kwargs and getattr(self, "_last", None) is not None:
            # Autoreset by the vector env (it calls reset() without arguments) after this episode ended: stay
            # frozen. A new batch resets with seed=..., which starts a real episode.
            obs, _, _, _, info = self._last
            return obs, info
        self._last = None
        return self.env.reset(**kwargs)

    @property
    def _init_state_id(self):  # the eval rollout reads and advances it through the wrapper chain
        return self.env._init_state_id

    @_init_state_id.setter
    def _init_state_id(self, value):
        self.env._init_state_id = value

    def step(self, action):
        if getattr(self, "_last", None) is not None:
            return self._last
        transition = self.env.step(action)
        if transition[2] or transition[3]:
            self._last = transition
        return transition


def make_env(cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the config.

    Args:
        cfg (EnvConfig): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        gym.vector.VectorEnv: The parallelized gym.env instance.
    """
    if n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    package_name = f"gym_{cfg.type}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    gym_handle = f"{package_name}/{cfg.task}"

    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    # FFI26 patch: finished envs return their last transition instead of simulating on (see _FreezeWhenDone).
    freeze = os.environ.get("CLARE_FREEZE_DONE", "1") != "0"

    def make_one():
        one = gym.make(gym_handle, disable_env_checker=True, **cfg.gym_kwargs)
        if freeze:
            # Insert right above the raw env, below gym.make's wrappers (TimeLimit, OrderEnforcing): the eval
            # rollout reads env.envs[i].env.env._init_state_id, so the wrapper depth must stay the same.
            parent = one
            while isinstance(parent.env, gym.Wrapper):
                parent = parent.env
            parent.env = _FreezeWhenDone(parent.env)
        return one

    env = env_cls([make_one for _ in range(n_envs)])

    return env
