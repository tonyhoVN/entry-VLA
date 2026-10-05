#!/usr/bin/env python3
"""Single-arm demo generator for the PerAct2 RLBench fork (run in the ``peract2`` env).

The fork's ``tools/dataset_generator.py`` still uses the old per-camera
``ObservationConfig`` attributes and ``obs.<camera>_rgb`` fields; the fork moved
cameras to ``ObservationConfig.camera_configs`` and ``obs.perception_data``. This
follows ``tools/dataset_generator_bimanual.py`` with ``robot_setup='panda'``.

Output (upstream single-arm layout):
  <save_path>/<task>/variation<V>/episodes/episode<N>/{low_dim_obs.pkl, variation_number.pkl,
  variation_descriptions.pkl, <cam>_rgb/rgb_0000.png, <cam>_depth/depth_0000.png, <cam>_mask/mask_0000.png}
Depth PNGs are RGB-packed linear buffers (DEPTH_SCALE = 2**24 - 1, depth_in_meters=False).
Demos are scripted waypoint/motion-planning rollouts, not teleoperation.
"""

import argparse
import logging
import os
import pickle

import numpy as np
from PIL import Image

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import (DEPTH_SCALE, EPISODE_FOLDER, EPISODES_FOLDER, LOW_DIM_PICKLE,
                                   VARIATION_DESCRIPTIONS, VARIATION_NUMBER, VARIATIONS_FOLDER)
from rlbench.backend.exceptions import BoundaryError, InvalidActionError, TaskEnvironmentError, WaypointError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig

# Vision sensors in rlbench/task_design.ttt (cam_<name> and cam_<name>_mask).
CAMERAS = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist", "front"]


class JointVelocityCompat(JointVelocity):
    # The fork's MoveArmThenGripper passes ignore_collisions; only planning modes accept it.
    def action(self, scene, action, ignore_collisions=None):
        super().action(scene, action)


def _patch_unimanual_release():
    # Scene._handle_extensions_strings calls robot.release_gripper(name) for waypoint
    # 'open_gripper()' extensions, but the fork's UnimanualRobot.release_gripper takes no name.
    from rlbench.backend.robot import UnimanualRobot
    UnimanualRobot.release_gripper = lambda self, name=None: self.gripper.release()


def save_demo(demo, episode_path, variation):
    for i, obs in enumerate(demo):
        for cam in CAMERAS:
            for dtype in ("rgb", "depth", "mask"):
                data = obs.perception_data.get(f"{cam}_{dtype}")
                if data is None:
                    continue
                if dtype == "rgb":
                    img = Image.fromarray(data)
                elif dtype == "depth":
                    img = utils.float_array_to_rgb_image(data, scale_factor=DEPTH_SCALE)
                else:
                    img = Image.fromarray((data * 255).astype(np.uint8))
                out = os.path.join(episode_path, f"{cam}_{dtype}")
                os.makedirs(out, exist_ok=True)
                img.save(os.path.join(out, f"{dtype}_{i:04d}.png"))
        obs.perception_data.clear()
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)
    with open(os.path.join(episode_path, VARIATION_NUMBER), "wb") as f:
        pickle.dump(variation, f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="task file name in rlbench/tasks, e.g. remove_cups")
    p.add_argument("--save_path", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--variation", type=int, default=0, help="variation index; -1 samples one per episode")
    p.add_argument("--image_size", type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    p.add_argument("--attempts", type=int, default=10, help="retries per episode on planning failures")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(args.seed)
    _patch_unimanual_release()

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.camera_configs = {
        cam: CameraConfig(image_size=tuple(args.image_size), depth_in_meters=False, masks_as_one_channel=False)
        for cam in CAMERAS}

    env = Environment(action_mode=MoveArmThenGripper(JointVelocityCompat(), Discrete()),
                      obs_config=obs_config, robot_setup="panda", headless=args.headless)
    env.launch()
    try:
        task_env = env.get_task(task_file_to_task_class(args.task, False))
        n_var = task_env.variation_count()
        if args.variation >= n_var:
            raise SystemExit(f"{args.task} has {n_var} variations; got --variation {args.variation}")
        logging.info("%s: %d variations", task_env.get_name(), n_var)

        for ex in range(args.episodes):
            for attempt in range(args.attempts):
                variation = np.random.randint(n_var) if args.variation < 0 else args.variation
                try:
                    task_env.set_variation(variation)
                    descriptions, _ = task_env.reset()
                    demo, = task_env.get_demos(amount=1, live_demos=True)
                    break
                except (BoundaryError, WaypointError, InvalidActionError, TaskEnvironmentError) as e:
                    logging.warning("episode %d attempt %d failed: %s", ex, attempt, e)
            else:
                raise SystemExit(f"episode {ex}: all {args.attempts} attempts failed")
            ep_path = os.path.join(args.save_path, task_env.get_name(), VARIATIONS_FOLDER % variation,
                                   EPISODES_FOLDER, EPISODE_FOLDER % ex)
            os.makedirs(ep_path, exist_ok=True)
            save_demo(demo, ep_path, variation)
            with open(os.path.join(ep_path, VARIATION_DESCRIPTIONS), "wb") as f:
                pickle.dump(descriptions, f)
            logging.info("saved %s (%d frames)", ep_path, len(demo))
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
