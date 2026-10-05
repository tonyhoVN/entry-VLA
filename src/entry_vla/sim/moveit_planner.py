"""Optional MoveIt 2 planning backend via ``moveit_py`` (ROS 2). User-run, opt-in.

The default simulator planner is ``entry_vla.jointspace_ik_interp/v2`` (not MoveIt).
Real-system claims require this integration (or an equivalent MoveIt pipeline).
Nothing here has been executed: ROS 2 / MoveIt are not installed in this
workspace. API used (moveit_py, ROS 2 Humble+): ``MoveItPy(node_name=...)``,
``get_planning_component(group)``, ``set_start_state(robot_state=...)``,
``set_goal_state(robot_state=...)``, ``plan()`` -> result with ``.trajectory``,
``RobotState.set_joint_group_positions``. Verify against the installed version.

Bimanual goals are expressed as a joint-space goal for a combined planning
group (IK per arm first, seeded from the current configuration), which keeps
both arms in one collision-checked plan.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..sim.contracts import ARMS, MissingDependencyError
from .preparation import Plan

MOVEIT_PLANNER_ID = "moveit_py"


class MoveItPlanner:  # pragma: no cover - requires ROS 2 + MoveIt
    planner_id = MOVEIT_PLANNER_ID

    def __init__(self, arms, group: str = "both_arms", joint_order: Optional[Dict[str, list]] = None,
                 node_name: str = "entry_vla_planner", fk_check_tol_m: float = 0.005):
        try:
            from moveit.core.robot_state import RobotState
            from moveit.planning import MoveItPy
        except ImportError as e:
            raise MissingDependencyError(f"moveit_py unavailable ({e}); source a ROS 2 + MoveIt 2 workspace") from e
        self.RobotState = RobotState
        self.moveit = MoveItPy(node_name=node_name)
        self.group = group
        self.pc = self.moveit.get_planning_component(group)
        self.model = self.moveit.get_robot_model()
        self.arms = arms            # entry_vla kinematic models for IK seeding/consistency checks
        self.joint_order = joint_order or {"left": list(range(7)), "right": list(range(7, 14))}
        self.tol = fk_check_tol_m

    def plan(self, q_start: Dict[str, np.ndarray], T_goal: Dict[str, np.ndarray]) -> Plan:
        import time
        t0 = time.perf_counter()
        q_goal = {}
        for a in ARMS:
            q, ok, info = self.arms[a].ik(T_goal[a], q_init=q_start[a])
            if not ok:
                return Plan(False, f"{a}:ik_failed", planning_time_s=time.perf_counter() - t0, planner=MOVEIT_PLANNER_ID)
            q_goal[a] = q
        start, goal = self.RobotState(self.model), self.RobotState(self.model)
        start.set_joint_group_positions(self.group, np.concatenate([q_start["left"], q_start["right"]]))
        goal.set_joint_group_positions(self.group, np.concatenate([q_goal["left"], q_goal["right"]]))
        self.pc.set_start_state(robot_state=start)
        self.pc.set_goal_state(robot_state=goal)
        res = self.pc.plan()
        if not res:
            return Plan(False, "moveit:no_plan", planning_time_s=time.perf_counter() - t0, planner=MOVEIT_PLANNER_ID)
        msg = res.trajectory.get_robot_trajectory_msg().joint_trajectory
        Q = np.array([p.positions for p in msg.points], dtype=np.float64)
        q_path = {a: Q[:, self.joint_order[a]] for a in ARMS}
        T_path = {a: np.stack([self.arms[a].fk(q) for q in q_path[a]]) for a in ARMS}
        for a in ARMS:   # the MoveIt model and the entry_vla model must agree on the final TCP pose
            err = float(np.linalg.norm(T_path[a][-1][:3, 3] - T_goal[a][:3, 3]))
            if err > self.tol:
                return Plan(False, f"moveit:model_mismatch({a},{err:.4f}m)", planning_time_s=time.perf_counter() - t0,
                            planner=MOVEIT_PLANNER_ID)
        p = Plan(True, q_path=q_path, T_path=T_path, planning_time_s=time.perf_counter() - t0, planner=MOVEIT_PLANNER_ID)
        p.reason = "moveit"
        return p
