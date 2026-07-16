"""Generic on-device data collector for gxm environments.

``Collector`` runs ``num_envs`` parallel environments for ``num_steps`` steps
per :meth:`Collector.collect` call and stacks whatever the ``extractor``
records into one batch pytree with leading axes ``(num_steps, num_envs)``.
The whole loop is a ``jax.lax.scan``, so ``jax.jit(collector.collect)`` is the
intended usage. State is carried in a :class:`CollectorState`, so successive
calls stream contiguous experience — feed each batch straight into a gradient
step and keep looping.

The three swappable pieces:

- ``env`` — any ``gxm.Environment`` (wrappers included).
- ``policy`` — a :class:`policies.Policy`: ``policy.init(key)`` builds the
  initial ``policy_state`` from a ``(num_envs,)`` batch of keys (called by
  :meth:`Collector.init`), and ``policy.apply(key, policy_state, obs)`` is
  called once per step on the full ``(num_envs, ...)`` observation batch.
  See :mod:`policies` for Random/Ensemble/Mixture implementations.
- ``extractor`` — ``extractor(env_state, timestep, action) -> dict`` deciding
  what each transition contributes to the batch. See :mod:`extractors`.
"""

import dataclasses
from typing import Any, Callable

import jax
from flax import struct
from gxm import Environment, Timestep
from gxm.typing import Key, PolicyState, PyTree

from .policies import Policy

Extractor = Callable[[PyTree, Timestep, PyTree], dict[str, Any]]


@struct.dataclass
class CollectorState:
    env_state: PyTree
    timestep: Timestep
    policy_state: PolicyState


class Collector:
    def __init__(
        self,
        env: Environment,
        policy: Policy,
        extractor: Extractor,
        num_envs: int,
        num_steps: int,
    ):
        self.env = env
        self.policy = policy
        self.extractor = extractor
        self.num_envs = num_envs
        self.num_steps = num_steps

    def init(self, key: Key) -> CollectorState:
        key_env, key_policy = jax.random.split(key)
        env_state, timestep = jax.vmap(self.env.init)(jax.random.split(key_env, self.num_envs))
        return CollectorState(
            env_state, timestep, self.policy.init(jax.random.split(key_policy, self.num_envs))
        )

    def collect(self, key: Key, state: CollectorState) -> tuple[CollectorState, dict[str, Any]]:
        def body(carry, key):
            env_state, timestep = carry
            key_policy, key_step = jax.random.split(key)
            action = self.policy.apply(key_policy, state.policy_state, timestep.next_obs)
            env_state, timestep = jax.vmap(self.env.step)(
                jax.random.split(key_step, self.num_envs), env_state, action
            )
            return (env_state, timestep), self.extractor(env_state, timestep, action)

        keys = jax.random.split(key, self.num_steps)
        (env_state, timestep), batch = jax.lax.scan(
            body, (state.env_state, state.timestep), keys
        )
        return dataclasses.replace(state, env_state=env_state, timestep=timestep), batch
