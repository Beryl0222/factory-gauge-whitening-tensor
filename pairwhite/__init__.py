"""缺测计量通道的成对加权统计与分组白化发布。

- :class:`PairwiseAccumulator` 流式累积组内通道对的充分统计量，
  支持重发去重与更正撤销，状态不随样本数平方增长。
- :class:`Publisher` 按交接分组做覆盖判定与双限制（谱范数、修正幅度）
  合规发布；无法同守时保留上一版，无上一版则标为不可用，组间互不影响。
- :class:`Pipeline` 把提交、发布、部署变换与审计报告串成可重放的流程。
"""
from .accumulator import (
    ChannelSummary,
    PairSummary,
    PairwiseAccumulator,
    SubmitOutcome,
)
from .common import TOL
from .pipeline import Pipeline, run_case
from .publisher import GroupSpec, PublishDecision, Publisher
from .whitening import WhiteningOutcome, symmetric_whitening

__all__ = [
    "ChannelSummary",
    "GroupSpec",
    "PairSummary",
    "PairwiseAccumulator",
    "Pipeline",
    "PublishDecision",
    "Publisher",
    "SubmitOutcome",
    "TOL",
    "WhiteningOutcome",
    "run_case",
    "symmetric_whitening",
]
