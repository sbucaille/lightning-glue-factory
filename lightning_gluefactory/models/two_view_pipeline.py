"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from lightning_gluefactory.models.base_model import BaseModel


class TwoViewPipeline(BaseModel):
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models

    def __init__(
            self,
            allow_no_extract: bool,
            run_gt_in_forward: bool,
            extractor=None,
            matcher=None,
            filter=None,
            solver=None,
            ground_truth=None,
            **kwargs
    ):
        self.allow_no_extract = allow_no_extract
        self.run_gt_in_forward = run_gt_in_forward
        super().__init__(**kwargs)
        self.extractor = extractor
        self.matcher = matcher
        self.filter = filter
        self.solver = solver
        self.ground_truth = ground_truth

    def _init(self):
        pass

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.allow_no_extract
        if self.extractor and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.extractor and not self.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def _forward(self, data):
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        if self.matcher:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.filter:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.solver:
            pred = {**pred, **self.solver({**data, **pred})}

        if self.ground_truth and self.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.ground_truth and not self.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.__getattribute__(k).keys():
                apply = self.conf[k].apply_loss
            if self.__getattribute__(k) and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
