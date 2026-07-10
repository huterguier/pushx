from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from gxm import Dynamics, DynamicsState, Step
from gxm.spaces import Box
from gxm.typing import Key
from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.scene import add_circle_to_scene, add_fixed_joint_to_scene, add_rectangle_to_scene
from jax2d.sim_state import SimParams, SimState, StaticSimParams

from pushx.shapes import Assembly

SCENE_SIZE = 2.0
PUSHER_RADIUS = 0.065
_N_STATIC = 4  # jax2d's num_static_fixated_polys


@jax.tree_util.register_dataclass
@dataclass
class PushState(DynamicsState):
    sim_state: SimState
    goal_pos: jnp.ndarray   # (2,)
    goal_angle: jnp.ndarray  # ()


class PushModel(Dynamics[PushState, Step]):
    def __init__(self, assembly: Assembly):
        self._assembly = assembly
        n_strokes = len(assembly.strokes)
        n_welds = len(assembly.welds)
        self._static = StaticSimParams(
            num_polygons=_N_STATIC + n_strokes,
            num_circles=2,  # jax2d requires at least 2 circle slots; we only use 1 (the pusher)
            num_joints=max(n_welds, 1),  # jax2d requires at least 1 joint slot
            num_thrusters=1,
            max_polygon_vertices=4,
        )
        self._params = SimParams(dt=1 / 60)
        self._engine = PhysicsEngine(self._static)
        self._base_sim_state, self._stroke_indices, self._pusher_idx = self._build_scene()

        goal_angle_bound = jnp.pi / assembly.symmetry_order
        self.observation_space = Box(
            low=jnp.array([0.0, 0.0, -jnp.pi, 0.0, 0.0, 0.0, 0.0, -goal_angle_bound]),
            high=jnp.array([SCENE_SIZE, SCENE_SIZE, jnp.pi, SCENE_SIZE, SCENE_SIZE, SCENE_SIZE, SCENE_SIZE, goal_angle_bound]),
            shape=(8,),
        )
        self.action_space = Box(
            low=jnp.zeros(2),
            high=jnp.full(2, SCENE_SIZE),
            shape=(2,),
        )

    def _build_scene(self):
        sim_state = create_empty_sim(
            self._static,
            add_floor=False,
            add_walls_and_ceiling=False,
            scene_size=SCENE_SIZE,
        )
        sim_state = sim_state.replace(gravity=jnp.zeros(2))
        center = jnp.full(2, SCENE_SIZE / 2)

        stroke_indices = []
        for stroke in self._assembly.strokes:
            pos = center + jnp.array(stroke.offset)
            sim_state, (idx, _) = add_rectangle_to_scene(
                sim_state,
                self._static,
                position=pos,
                dimensions=jnp.array(stroke.size),
                rotation=stroke.angle,
                friction=0.8,
                restitution=0.0,
            )
            stroke_indices.append(idx)

        for weld in self._assembly.welds:
            ref_angle = self._assembly.strokes[weld.b].angle - self._assembly.strokes[weld.a].angle
            sim_state, joint_idx = add_fixed_joint_to_scene(
                sim_state,
                self._static,
                a_index=stroke_indices[weld.a],
                b_index=stroke_indices[weld.b],
                a_relative_pos=jnp.array(weld.a_relative_pos),
                b_relative_pos=jnp.array(weld.b_relative_pos),
            )
            sim_state = sim_state.replace(
                joint=sim_state.joint.replace(
                    rotation=sim_state.joint.rotation.at[joint_idx].set(ref_angle)
                )
            )

        sim_state, (pusher_idx, _) = add_circle_to_scene(
            sim_state,
            self._static,
            position=jnp.array([SCENE_SIZE * 0.7, SCENE_SIZE / 2]),
            radius=PUSHER_RADIUS,
            density=5.0,
            friction=0.8,
            restitution=0.0,
        )

        return sim_state, stroke_indices, pusher_idx

    def _obs(self, sim_state: SimState, goal_pos: jnp.ndarray, goal_angle: jnp.ndarray) -> jnp.ndarray:
        root_pos = sim_state.polygon.position[self._stroke_indices[0]]
        # Report assembly angle (= world angle - stroke_0 offset) so it's comparable to goal_angle
        assembly_angle = sim_state.polygon.rotation[self._stroke_indices[0]] - self._assembly.strokes[0].angle
        pusher_pos = sim_state.circle.position[self._pusher_idx]
        return jnp.concatenate([root_pos, assembly_angle[None], pusher_pos, goal_pos, goal_angle[None]])

    def _randomise(self, key: Key) -> SimState:
        key_pos, key_angle, key_pusher = jax.random.split(key, 3)
        margin = 0.3
        lo, hi = margin, SCENE_SIZE - margin

        assembly_pos = jax.random.uniform(key_pos, (2,), minval=lo, maxval=hi)
        assembly_angle = jax.random.uniform(key_angle, (), minval=-jnp.pi, maxval=jnp.pi)
        pusher_pos = jax.random.uniform(key_pusher, (2,), minval=lo, maxval=hi)

        cos_a, sin_a = jnp.cos(assembly_angle), jnp.sin(assembly_angle)
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])

        sim = self._base_sim_state
        positions = sim.polygon.position
        rotations = sim.polygon.rotation
        for stroke, idx in zip(self._assembly.strokes, self._stroke_indices):
            world_pos = assembly_pos + rot @ jnp.array(stroke.offset)
            world_angle = assembly_angle + stroke.angle
            positions = positions.at[idx].set(world_pos)
            rotations = rotations.at[idx].set(world_angle)

        poly = sim.polygon.replace(
            position=positions,
            rotation=rotations,
            velocity=jnp.zeros_like(sim.polygon.velocity),
            angular_velocity=jnp.zeros_like(sim.polygon.angular_velocity),
        )
        circle = sim.circle.replace(
            position=sim.circle.position.at[self._pusher_idx].set(pusher_pos),
            velocity=jnp.zeros_like(sim.circle.velocity),
            angular_velocity=jnp.zeros_like(sim.circle.angular_velocity),
        )
        joint = sim.joint.replace(
            acc_impulse=jnp.zeros_like(sim.joint.acc_impulse),
            acc_r_impulse=jnp.zeros_like(sim.joint.acc_r_impulse),
        )
        return sim.replace(polygon=poly, circle=circle, joint=joint)

    def _random_goal(self, key: Key) -> tuple[jnp.ndarray, jnp.ndarray]:
        key_pos, key_angle = jax.random.split(key)
        margin = 0.4
        goal_pos = jax.random.uniform(key_pos, (2,), minval=margin, maxval=SCENE_SIZE - margin)
        # Sample from canonical range [−π/k, π/k] so goal_angle is unambiguous in raw obs
        half = jnp.pi / self._assembly.symmetry_order
        goal_angle = jax.random.uniform(key_angle, (), minval=-half, maxval=half)
        return goal_pos, goal_angle

    def init(self, key: Key) -> tuple[PushState, Step]:
        key_sim, key_goal = jax.random.split(key)
        sim_state = self._randomise(key_sim)
        goal_pos, goal_angle = self._random_goal(key_goal)
        state = PushState(sim_state=sim_state, goal_pos=goal_pos, goal_angle=goal_angle)
        return state, Step(
            next_obs=self._obs(sim_state, goal_pos, goal_angle),
            action=jnp.zeros(2),
            info={},
        )

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Step]:
        return self.init(key)  # state is ignored; full re-randomisation on every reset

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Step]:
        pusher_pos = state.sim_state.circle.position[self._pusher_idx]
        target_vel = jnp.clip((action - pusher_pos) * 5.0, -3.0, 3.0)

        circle = state.sim_state.circle.replace(
            velocity=state.sim_state.circle.velocity.at[self._pusher_idx].set(target_vel)
        )
        sim = state.sim_state.replace(circle=circle)
        new_sim, _ = self._engine.step(
            sim,
            self._params,
            jnp.zeros(self._static.num_joints + self._static.num_thrusters),
        )

        damping = 0.6
        poly = new_sim.polygon.replace(
            velocity=new_sim.polygon.velocity * damping,
            angular_velocity=new_sim.polygon.angular_velocity * damping,
        )
        new_sim = new_sim.replace(polygon=poly)

        new_state = PushState(
            sim_state=new_sim, goal_pos=state.goal_pos, goal_angle=state.goal_angle
        )
        return new_state, Step(
            next_obs=self._obs(new_sim, state.goal_pos, state.goal_angle),
            action=action,
            info={},
        )

    def render(self, state: PushState, width: int = 64, height: int = 64) -> jnp.ndarray:
        from pushx.render import _render_pixels
        return _render_pixels(state, self._assembly, self._stroke_indices, self._pusher_idx, width, height)
