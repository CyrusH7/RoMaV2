"""Validated backend settings independent of argparse and COLMAP objects."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineOptions:
    cell_size: float = 1.0
    min_matches: int = 15
    min_match_score: float = 0.0
    geometric_max_error: float = 4.0
    geometric_max_trials: int = 20000
    geometric_min_inlier_ratio: float = 0.1
    fix_intrinsics: bool = False
    seed: int = 0
    use_intrinsics_prior: bool = False
    use_pose_prior: bool = False
    prior_iterations: int = 3
    joint_prior_ba: bool = False
    camera_model: str | None = None
    num_threads: int = 4

    fix_poses: bool = False
    init_from_poses: bool = False
    single_camera: bool = False
    refine_initial_pose: bool | str = False

    @property
    def triangulate_from_priors(self):
        return self.fix_poses or self.init_from_poses

    def __post_init__(self):
        if self.joint_prior_ba and (
            not self.use_pose_prior or not self.fix_intrinsics or self.fix_poses
        ):
            raise ValueError(
                "Joint prior BA requires variable poses, pose priors and fixed calibration"
            )
        if self.fix_poses and self.init_from_poses:
            raise ValueError("Choose either fixed poses or initialization from poses")
        if self.refine_initial_pose not in (
            False,
            True,
            "rotation",
            "joint",
            "landmarks",
            "graph",
        ):
            raise ValueError("Unknown initial pose refinement mode")
        if self.refine_initial_pose and not self.init_from_poses:
            raise ValueError("Sampson refinement requires --init-from-poses")
        if self.triangulate_from_priors and not self.fix_intrinsics:
            raise ValueError(
                "Triangulation from prior poses requires --fix-intrinsics and known calibration"
            )
        if not math.isfinite(self.cell_size) or self.cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")
        if self.min_matches < 8:
            raise ValueError("min_matches must be at least 8")
        if (
            not math.isfinite(self.min_match_score)
            or not 0 <= self.min_match_score <= 1
        ):
            raise ValueError("min_match_score must be in [0, 1]")
        if not math.isfinite(self.geometric_max_error) or self.geometric_max_error <= 0:
            raise ValueError("geometric_max_error must be positive and finite")
        if not 0 < self.geometric_min_inlier_ratio <= 1:
            raise ValueError("geometric_min_inlier_ratio must be in (0, 1]")
        if self.geometric_max_trials < 100:
            raise ValueError("geometric_max_trials must be at least 100")
        if min(self.num_threads, self.prior_iterations) < 1:
            raise ValueError("num_threads and prior_iterations must be positive")
        if self.camera_model not in (None, "PINHOLE", "SIMPLE_RADIAL"):
            raise ValueError("camera_model must be PINHOLE or SIMPLE_RADIAL")
