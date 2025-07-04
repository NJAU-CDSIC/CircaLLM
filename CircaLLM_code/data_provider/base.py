from dataclasses import dataclass

import numpy.typing as npt


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    logits: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    embeddings1: npt.NDArray = None
    embeddings2: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False
