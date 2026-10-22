from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from gxm import Dynamics, DynamicsState, Step
from gxm.spaces import Box
from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.scene import (
    add_circle_to_scene,
    add_fixed_joint_to_scene,
    add_rectangle_to_scene,
)
from jax2d.sim_state import SimParams, SimState, StaticSimParams

from pushx.shapes import Assembly
from pushx.typing import Array, Key

SCENE_SIZE = 2.0
_NUM_STATIC = 4  # jax2d's num_static_fixated_polys

# swm/PushT's pymunk scene lives in a 512px window; this is the scale factor that maps
# its pixel-space geometry onto pushx's SCENE_SIZE world, so shapes sized from PushT's
# pixel constants (see letters.py's T) and the pusher radius below land at the same
# proportions.
_PUSHT_PX_SCALE = SCENE_SIZE / 512.0
# PushT agent: base_radius=0.375 * scale=40 = 15px
PUSHER_RADIUS = 15.0 * _PUSHT_PX_SCALE

# PushT bounds its arena with 4 static wall segments at its window's [5, 506]px margins
# (radius=2, so 4px thick). Module-level because `render` draws them too -- PushT's
# `space.debug_draw` renders its walls into every frame, so a scene without them is a
# visibly different picture. Defined above the `render` import, which consumes them.
WALL_LO = 5.0 * _PUSHT_PX_SCALE
WALL_HI = 506.0 * _PUSHT_PX_SCALE
WALL_MID = (WALL_LO + WALL_HI) / 2
WALL_SPAN = WALL_HI - WALL_LO
WALL_THICKNESS = 4.0 * _PUSHT_PX_SCALE

# Reset bounds, as [lo, hi] in world units, matching PushT's `variation_space`:
# block/goal `start_position` is U[100, 400]px and the agent's is U[50, 450]px on a
# 512px window, i.e. [0.391, 1.563] and [0.195, 1.758] here. Rounded to the nearest
# tenth, which is within 0.04 of PushT and keeps both ranges centred on the scene.
ASSEMBLY_MARGIN = 0.4  # -> [0.4, 1.6]
PUSHER_MARGIN = 0.2  # -> [0.2, 1.8]
GOAL_MARGIN = 0.4  # -> [0.4, 1.6]; PushT samples the goal from the block's own space

from pushx.render import _render_pixels  # noqa: E402


@jax.tree_util.register_dataclass
@dataclass
class PushState(DynamicsState):
    sim_state: SimState
    goal_pos: Array  # (2,)
    goal_angle: Array  # ()


def _derived():
    return field(init=False, repr=False, compare=False)


@dataclass(frozen=True)
class PushDynamics(Dynamics[PushState, Step]):
    """``goal`` fixes the target pose as ``((x, y), angle)`` for every episode;
    ``None`` samples a new one per episode, as PushT does. The angle is an
    assembly angle and is wrapped into the canonical ``[-pi/k, pi/k]`` the
    observation uses, ``k`` being the assembly's symmetry order.

    ``spawn`` likewise fixes the initial state as ``(block_pos, block_angle,
    pusher_pos)``; ``None`` samples all three uniformly over the arena, as
    PushT does. A fixed spawn is jittered by ``spawn_jitter = (distance,
    angle)``: both positions get a uniform offset within ``±distance`` per
    axis and the block angle one within ``±angle`` radians."""

    assembly: Assembly
    goal: tuple[tuple[float, float], float] | None = None
    spawn: tuple[tuple[float, float], float, tuple[float, float]] | None = None
    spawn_jitter: tuple[float, float] = (0.1, 0.3)

    observation_space: Box = _derived()
    action_space: Box = _derived()
    _static: StaticSimParams = _derived()
    _params: SimParams = _derived()
    _engine: PhysicsEngine = _derived()
    _base_sim_state: SimState = _derived()
    _stroke_indices: list[int] = _derived()
    _pusher_idx: int = _derived()

    def __post_init__(self):
        num_strokes = len(self.assembly.strokes)
        num_welds = len(self.assembly.welds)
        static = StaticSimParams(
            num_polygons=_NUM_STATIC + num_strokes,
            # jax2d requires at least 2 circle slots; we only use 1 (the pusher)
            num_circles=2,
            num_joints=max(num_welds, 1),  # jax2d requires at least 1 joint slot
            num_thrusters=1,
            max_polygon_vertices=4,
        )
        object.__setattr__(self, "_static", static)
        # matches PushT's physics dt exactly (see ActionRepeat)
        object.__setattr__(self, "_params", SimParams(dt=0.01))
        object.__setattr__(self, "_engine", PhysicsEngine(static))
        base_sim_state, stroke_indices, pusher_idx = self._build_scene()
        object.__setattr__(self, "_base_sim_state", base_sim_state)
        object.__setattr__(self, "_stroke_indices", stroke_indices)
        object.__setattr__(self, "_pusher_idx", pusher_idx)

        goal_angle_bound = jnp.pi / self.assembly.symmetry_order
        observation_space = Box(
            low=jnp.array([0.0, 0.0, -jnp.pi, 0.0, 0.0, 0.0, 0.0, -goal_angle_bound]),
            high=jnp.array(
                [
                    SCENE_SIZE,
                    SCENE_SIZE,
                    jnp.pi,
                    SCENE_SIZE,
                    SCENE_SIZE,
                    SCENE_SIZE,
                    SCENE_SIZE,
                    goal_angle_bound,
                ]
            ),
            shape=(8,),
        )
        action_space = Box(low=jnp.zeros(2), high=jnp.full(2, SCENE_SIZE), shape=(2,))
        object.__setattr__(self, "observation_space", observation_space)
        object.__setattr__(self, "action_space", action_space)

    def _build_scene(self):
        sim_state = create_empty_sim(
            self._static,
            add_floor=False,
            add_walls_and_ceiling=False,
            scene_size=SCENE_SIZE,
        )
        sim_state = sim_state.replace(gravity=jnp.zeros(2))
        center = jnp.full(2, SCENE_SIZE / 2)

        # Frictionless, like everything else here -- see the stroke loop's friction
        # comment. `_NUM_STATIC` was already reserved for exactly this; walls fill it
        # first so strokes/joints/pusher indices are unaffected. Geometry is the
        # module-level WALL_* so `render` draws the same walls the physics uses.
        long_side = WALL_SPAN + WALL_THICKNESS
        walls = [
            (jnp.array([WALL_LO, WALL_MID]), jnp.array([WALL_THICKNESS, long_side])),
            (jnp.array([WALL_HI, WALL_MID]), jnp.array([WALL_THICKNESS, long_side])),
            (jnp.array([WALL_MID, WALL_LO]), jnp.array([long_side, WALL_THICKNESS])),
            (jnp.array([WALL_MID, WALL_HI]), jnp.array([long_side, WALL_THICKNESS])),
        ]
        for wall_pos, wall_dims in walls:
            sim_state, _ = add_rectangle_to_scene(
                sim_state,
                self._static,
                position=wall_pos,
                dimensions=wall_dims,
                friction=0.0,
                restitution=0.0,
                fixated=True,
            )

        stroke_indices = []
        for stroke in self.assembly.strokes:
            pos = center + jnp.array(stroke.offset)
            # PushT builds each stroke as a separate pymunk Poly with mass=1 fixed
            # (moment_for_poly(1, vertices)), regardless of the piece's area -- not a
            # uniform-density body. Match that: give every stroke equal nominal mass
            # by scaling density inversely with its own area (mass = density * area).
            area = stroke.size[0] * stroke.size[1]
            sim_state, (idx, _) = add_rectangle_to_scene(
                sim_state,
                self._static,
                position=pos,
                dimensions=jnp.array(stroke.size),
                rotation=stroke.angle,
                density=1.0 / area,
                # PushT sets `body.friction` (not `shape.friction`) on its
                # blocks/agent/walls, which pymunk's Body has no such attribute
                # for -- it's silently ignored, so every contact in PushT is
                # actually frictionless (shape friction defaults to 0.0). Matched
                # here rather than the shape.friction=1 the code visually appears
                # to ask for.
                friction=0.0,
                restitution=0.0,
            )
            stroke_indices.append(idx)

        for weld in self.assembly.welds:
            ref_angle = (
                self.assembly.strokes[weld.b].angle
                - self.assembly.strokes[weld.a].angle
            )
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
            # PushT's agent is frictionless too -- see the stroke loop's comment above
            friction=0.0,
            restitution=0.0,
            # kinematic: infinite mass, so contact can't knock the pusher off its
            # commanded velocity
            fixated=True,
        )

        return sim_state, stroke_indices, pusher_idx

    def _obs(self, sim_state: SimState, goal_pos: Array, goal_angle: Array) -> Array:
        root_pos = sim_state.polygon.position[self._stroke_indices[0]]
        # Report assembly angle (= world angle - stroke_0 offset) so it's comparable
        # to goal_angle
        assembly_angle = (
            sim_state.polygon.rotation[self._stroke_indices[0]]
            - self.assembly.strokes[0].angle
        )
        pusher_pos = sim_state.circle.position[self._pusher_idx]
        return jnp.concatenate(
            [root_pos, assembly_angle[None], pusher_pos, goal_pos, goal_angle[None]]
        )

    def _rigidify(self, position: Array, rotation: Array) -> tuple[Array, Array]:
        # PushT's block is a single rigid body carrying several shapes; here it is
        # several bodies held by jax2d's soft fixed joints, which the fixated pusher
        # can tear apart by crushing the block into a wall -- after which the solver
        # diverges to NaN. Snap the strokes back onto their mean rigid pose every step
        # (strokes have equal mass, so the plain mean preserves the center of mass).
        idx = jnp.array(self._stroke_indices)
        offsets = jnp.array([s.offset for s in self.assembly.strokes])
        angles = jnp.array([s.angle for s in self.assembly.strokes])
        estimates = rotation[idx] - angles
        spread = (estimates - estimates[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        angle = estimates[0] + jnp.mean(spread)
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        world_offsets = offsets @ jnp.array([[cos, -sin], [sin, cos]]).T
        origin = jnp.mean(position[idx] - world_offsets, axis=0)
        return position.at[idx].set(origin + world_offsets), rotation.at[idx].set(
            angle + angles
        )

    def _randomise(self, key: Key) -> SimState:
        key_pos, key_angle, key_pusher = jax.random.split(key, 3)

        # Separate margins: PushT samples the agent from a wider box than the block
        # (50..450px vs 100..400px), so a single shared margin never let the pusher
        # start outside the block's own band.
        if self.spawn is None:
            assembly_pos = jax.random.uniform(
                key_pos,
                (2,),
                minval=ASSEMBLY_MARGIN,
                maxval=SCENE_SIZE - ASSEMBLY_MARGIN,
            )
            assembly_angle = jax.random.uniform(
                key_angle, (), minval=-jnp.pi, maxval=jnp.pi
            )
            pusher_pos = jax.random.uniform(
                key_pusher,
                (2,),
                minval=PUSHER_MARGIN,
                maxval=SCENE_SIZE - PUSHER_MARGIN,
            )
        else:
            assembly_pos, assembly_angle, pusher_pos = (
                jnp.asarray(v, jnp.float32) for v in self.spawn
            )
            distance, angle = self.spawn_jitter
            assembly_pos = assembly_pos + jax.random.uniform(
                key_pos, (2,), minval=-distance, maxval=distance
            )
            assembly_angle = assembly_angle + jax.random.uniform(
                key_angle, (), minval=-angle, maxval=angle
            )
            pusher_pos = pusher_pos + jax.random.uniform(
                key_pusher, (2,), minval=-distance, maxval=distance
            )

        cos_a, sin_a = jnp.cos(assembly_angle), jnp.sin(assembly_angle)
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])

        sim = self._base_sim_state
        positions = sim.polygon.position
        rotations = sim.polygon.rotation
        for stroke, idx in zip(
            self.assembly.strokes, self._stroke_indices, strict=True
        ):
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

    def _goal_for_episode(self, key: Key) -> tuple[Array, Array]:
        if self.goal is None:
            return self._random_goal(key)
        goal_pos, goal_angle = self.goal
        half = jnp.pi / self.assembly.symmetry_order
        goal_angle = (jnp.float32(goal_angle) + half) % (2 * half) - half
        return jnp.asarray(goal_pos, jnp.float32).reshape(2), goal_angle

    def _random_goal(self, key: Key) -> tuple[Array, Array]:
        key_pos, key_angle = jax.random.split(key)
        goal_pos = jax.random.uniform(
            key_pos, (2,), minval=GOAL_MARGIN, maxval=SCENE_SIZE - GOAL_MARGIN
        )
        # Sample from canonical range [−π/k, π/k] so goal_angle is unambiguous in
        # raw obs
        half = jnp.pi / self.assembly.symmetry_order
        goal_angle = jax.random.uniform(key_angle, (), minval=-half, maxval=half)
        return goal_pos, goal_angle

    def init(self, key: Key) -> tuple[PushState, Step]:
        key_sim, key_goal = jax.random.split(key)
        sim_state = self._randomise(key_sim)
        goal_pos, goal_angle = self._goal_for_episode(key_goal)
        state = PushState(sim_state=sim_state, goal_pos=goal_pos, goal_angle=goal_angle)
        return state, Step(
            next_obs=self._obs(sim_state, goal_pos, goal_angle),
            action=jnp.zeros(2),
            info={},
        )

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Step]:
        del state  # full re-randomisation on every reset
        return self.init(key)

    def step(self, key: Key, state: PushState, action: Array) -> tuple[PushState, Step]:
        del key
        # PD control toward `action` (an absolute world-space target), matching
        # PushT's agent controller: k_p/k_v are time constants (1/s^2, 1/s) and so
        # are independent of SCENE_SIZE's coordinate scale -- these are the same
        # gains PushT uses on its 512px window. k_p=100, k_v=20 is critically
        # damped (zeta = k_v / (2*sqrt(k_p)) = 1).
        k_p, k_v = 100.0, 20.0
        pusher_pos = state.sim_state.circle.position[self._pusher_idx]
        pusher_vel = state.sim_state.circle.velocity[self._pusher_idx]
        accel = k_p * (action - pusher_pos) - k_v * pusher_vel
        new_vel = pusher_vel + accel * self._params.dt

        circle = state.sim_state.circle.replace(
            velocity=state.sim_state.circle.velocity.at[self._pusher_idx].set(new_vel)
        )
        sim = state.sim_state.replace(circle=circle)
        new_sim, _ = self._engine.step(
            sim,
            self._params,
            jnp.zeros(self._static.num_joints + self._static.num_thrusters),
        )

        # PushT runs its pymunk space at `damping = 0`, and pymunk scales velocity by
        # `damping ** dt` at the *start* of each step -- so the block enters every step
        # from rest, is moved by that step's contact impulse alone, and never coasts.
        # Zeroing here, after the engine has already integrated positions with the
        # post-impulse velocity, reproduces that ordering exactly. The pusher is
        # `fixated` and unaffected, as PushT's KINEMATIC agent is.
        #
        # This replaced a 0.736 per-step retention, which let the block carry momentum
        # between steps and was the larger half of pushx pushing the block ~1.2x
        # further per contact than PushT does.
        position, rotation = self._rigidify(
            new_sim.polygon.position, new_sim.polygon.rotation
        )
        poly = new_sim.polygon.replace(
            position=position,
            rotation=rotation,
            velocity=jnp.zeros_like(new_sim.polygon.velocity),
            angular_velocity=jnp.zeros_like(new_sim.polygon.angular_velocity),
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

    def render(self, state: PushState, width: int = 64, height: int = 64) -> Array:
        return _render_pixels(
            state, self.assembly, self._stroke_indices, self._pusher_idx, width, height
        )
