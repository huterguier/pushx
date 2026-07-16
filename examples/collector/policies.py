"""Composable policies for the collector.

A ``Policy`` is anything with two methods: ``init(key)`` builds the initial
``policy_state`` — the key's shape carries the batch, one key per env (a
scalar key means no leading batch axis) — and ``apply(key, policy_state,
obs)`` maps the full ``(num_envs, ...)`` observation batch to actions.
Construction-time data
(action spaces, checkpoints, sub-policies) lives on the instance; everything
that should be swappable between ``collect`` calls without retracing flows
through ``policy_state``. ``Ensemble`` and ``Mixture`` are combinators: their
``init`` assembles the combined state from the sub-policies' states, so state
structure always matches apply-time expectations by construction.

``apply`` keeps the ``gxm.typing.Policy`` signature, so ``policy.apply`` is a
valid gxm policy and a bare gxm policy adapts via ``Ensemble``/a trivial
wrapper.
"""

from typing import Protocol

import jax
import jax.numpy as jnp
from gxm.spaces import Box
from gxm.typing import Action, Key, Observation, PolicyState
from gxm.typing import Policy as PolicyFn


class Policy(Protocol):
    def init(self, key: Key) -> PolicyState: ...

    def apply(self, key: Key, policy_state: PolicyState, obs: Observation) -> Action: ...


class Random:
    """Uniform random actions over a ``Box`` action space. Stateless."""

    def __init__(self, action_space: Box):
        self.action_space = action_space

    def init(self, key: Key) -> PolicyState:
        return None

    def apply(self, key: Key, policy_state: PolicyState, obs: Observation) -> Action:
        num_envs = jax.tree.leaves(obs)[0].shape[0]
        return jax.random.uniform(
            key,
            (num_envs, *self.action_space.shape),
            minval=self.action_space.low,
            maxval=self.action_space.high,
        )


class Ensemble:
    """Drive the env batch with a stack of parameter sets for one policy fn.

    ``policy`` is a gxm-style callable ``(key, params, obs) -> action`` and
    ``states`` a pytree with leading axis = members (e.g. stacked
    checkpoints); ``init`` returns it as the ``policy_state``, so a fresh
    stack can be substituted between ``collect`` calls. Env ``i`` of
    ``num_envs`` is driven by member ``i * num_members // num_envs``, so a
    checkpoint sequence spreads evenly over the batch. Every member is
    evaluated on the full observation batch and the result gathered per env —
    for small networks this is cheaper than gathering per-env parameters.
    """

    def __init__(self, policy: PolicyFn, states: PolicyState):
        self.policy = policy
        self.states = states

    def init(self, key: Key) -> PolicyState:
        return self.states

    def apply(self, key: Key, policy_state: PolicyState, obs: Observation) -> Action:
        num_members = jax.tree.leaves(policy_state)[0].shape[0]
        num_envs = jax.tree.leaves(obs)[0].shape[0]
        member = jnp.arange(num_envs) * num_members // num_envs
        actions = jax.vmap(self.policy, in_axes=(0, 0, None))(
            jax.random.split(key, num_members), policy_state, obs
        )
        return jax.tree.map(
            lambda x: jnp.take_along_axis(
                x, member.reshape(1, num_envs, *([1] * (x.ndim - 2))), axis=0
            )[0],
            actions,
        )


class Mixture:
    """Partition the env batch into contiguous blocks, one policy per block.

    ``fractions`` must sum to 1; block ``i`` covers ``fractions[i]`` of the
    envs. The combined ``policy_state`` is a tuple of the sub-policies'
    states, assembled by ``init``. Every sub-policy runs on the full batch
    and the actions are selected per env, so all sub-policies must produce
    identically-shaped actions.
    """

    def __init__(self, policies: list[Policy], fractions: list[float]):
        assert len(policies) == len(fractions)
        self.policies = policies
        self.fractions = fractions

    def init(self, key: Key) -> PolicyState:
        fold_in = jax.random.fold_in
        for _ in range(key.ndim):  # fold_in takes a scalar key; lift over any batch shape
            fold_in = jax.vmap(fold_in, in_axes=(0, None))
        return tuple(policy.init(fold_in(key, i)) for i, policy in enumerate(self.policies))

    def apply(self, key: Key, policy_state: PolicyState, obs: Observation) -> Action:
        num_envs = jax.tree.leaves(obs)[0].shape[0]
        bounds = jnp.cumsum(jnp.array(self.fractions)) * num_envs
        block = jnp.searchsorted(bounds, jnp.arange(num_envs), side="right")
        actions = [
            policy.apply(k, state, obs)
            for policy, state, k in zip(
                self.policies, policy_state, jax.random.split(key, len(self.policies))
            )
        ]
        return jax.tree.map(
            lambda *xs: jnp.take_along_axis(
                jnp.stack(xs), block.reshape(1, num_envs, *([1] * (xs[0].ndim - 1))), axis=0
            )[0],
            *actions,
        )
