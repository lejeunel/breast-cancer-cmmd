from typing import Iterable, Optional, Tuple, TypeVar

import torch
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_update_input_check,
)
from torcheval.metrics.metric import Metric

TBinaryRecallAtFixedPrecision = TypeVar("TBinaryPrecisionAtFixedRecall")


def _binary_precision_at_fixed_recall_update_input_check(
    input: torch.Tensor, target: torch.Tensor, min_recall: float
) -> None:
    _binary_precision_recall_curve_update_input_check(input, target)
    if not isinstance(min_recall, float) or not (0 <= min_recall <= 1):
        raise ValueError(
            "Expected min_recall to be a float in the [0, 1] range"
            f" but got {min_recall}."
        )


@torch.jit.script
def _precision_at_recall(
    precision: torch.Tensor,
    recall: torch.Tensor,
    thresholds: torch.Tensor,
    min_recall: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_precision = torch.max(precision[recall >= min_recall])
    thresholds = torch.cat((thresholds, torch.tensor([-1.0], device=thresholds.device)))
    best_threshold = torch.max(thresholds[precision == max_precision])
    return max_precision, torch.abs(best_threshold)


def _binary_precision_at_fixed_recall_compute(
    input: torch.Tensor, target: torch.Tensor, min_recall: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    precision, recall, threshold = _binary_precision_recall_curve_compute(input, target)
    return _precision_at_recall(precision, recall, threshold, min_recall)


class BinaryPrecisionAtFixedRecall(Metric[Tuple[torch.Tensor, torch.Tensor]]):

    def __init__(
        self: TBinaryRecallAtFixedPrecision,
        *,
        min_recall: float,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self.min_recall = min_recall
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryRecallAtFixedPrecision,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryRecallAtFixedPrecision:
        input = input.to(self.device)
        target = target.to(self.device)

        _binary_precision_at_fixed_recall_update_input_check(
            input, target, self.min_recall
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryRecallAtFixedPrecision,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _binary_precision_at_fixed_recall_compute(
            torch.cat(self.inputs), torch.cat(self.targets), self.min_recall
        )[0]

    @torch.inference_mode()
    def merge_state(
        self: TBinaryRecallAtFixedPrecision,
        metrics: Iterable[TBinaryRecallAtFixedPrecision],
    ) -> TBinaryRecallAtFixedPrecision:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TBinaryRecallAtFixedPrecision) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
