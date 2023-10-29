from typing import Optional

from pydantic import BaseModel
from torch import Tensor


class CustomBaseModel(BaseModel):
	class Config:
		arbitrary_types_allowed = True


class TwoViewPipelineInput(CustomBaseModel):
	view0: Optional[Tensor]
	view1: Optional[Tensor]


class Input(CustomBaseModel):
	"""
	Matcher input data
	"""
	keypoints0: Tensor
	keypoints1: Tensor
	descriptors0: Tensor
	descriptors1: Tensor

	scales0: Optional[Tensor]
	scales1: Optional[Tensor]

	oris0: Optional[Tensor]
	oris1: Optional[Tensor]

	view0: Optional[Tensor]
	view1: Optional[Tensor]

	H_0to1: Optional[Tensor]
	H_1to0: Optional[Tensor]


class MatcherOutput(CustomBaseModel):
	matches0: Tensor
	matches1: Tensor
	matching_scores0: Tensor
	matching_scores1: Tensor
	ref_descriptors0: Tensor
	ref_descriptors1: Tensor
	log_assignment: Tensor
	prune0: Tensor
	prune1: Tensor
