from gluefactory.geometry.gt_generation import (
    gt_line_matches_from_homography,
    gt_matches_from_homography,
)
from lightning_gluefactory.models.base_model import BaseModel


class HomographyMatcher(BaseModel):

    required_data_keys = ["H_0to1"]

    def __init__(
            self,
            use_points: bool,
            th_positive: float,
            th_negative: float,
            use_lines: bool,
            n_line_sampled_pts: int,
            line_perp_dist_th: int,
            overlap_th: float,
            min_visibility_th: float,
            **kwargs
    ):
        self.use_points = use_points
        self.th_positive = th_positive
        self.th_negative = th_negative
        self.use_lines = use_lines
        self.n_line_sampled_pts = n_line_sampled_pts
        self.line_perp_dist_th = line_perp_dist_th
        self.overlap_th = overlap_th
        self.min_visibility_th = min_visibility_th
        super().__init__(**kwargs)

    def _init(self):
        # TODO (iago): Is this just boilerplate code?
        if self.use_points:
            self.required_data_keys += ["keypoints0", "keypoints1"]
        if self.use_lines:
            self.required_data_keys += [
                "lines0",
                "lines1",
                "valid_lines0",
                "valid_lines1",
            ]

    def _forward(self, data):
        result = {}
        if self.use_points:
            result = gt_matches_from_homography(
                data["keypoints0"],
                data["keypoints1"],
                data["H_0to1"],
                pos_th=self.th_positive,
                neg_th=self.th_negative,
            )
        if self.use_lines:
            line_assignment, line_m0, line_m1 = gt_line_matches_from_homography(
                data["lines0"],
                data["lines1"],
                data["valid_lines0"],
                data["valid_lines1"],
                data["view0"]["image"].shape,
                data["view1"]["image"].shape,
                data["H_0to1"],
                self.n_line_sampled_pts,
                self.line_perp_dist_th,
                self.overlap_th,
                self.min_visibility_th,
            )
            result["line_matches0"] = line_m0
            result["line_matches1"] = line_m1
            result["line_assignment"] = line_assignment
        return result
