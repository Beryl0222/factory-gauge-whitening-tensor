"""更正到部署的可重放管线：提交 -> 发布 -> 部署变换 -> 审计报告。"""
from __future__ import annotations

from collections.abc import Mapping

from .accumulator import PairwiseAccumulator, SubmitOutcome
from .publisher import GroupSpec, Publisher, pairs_for_groups


class Pipeline:
    """以通道名为界面串起累积器与发布器。"""

    def __init__(self, channels, groups):
        self.channels = list(channels)
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("通道名重复")
        self._index = {name: k for k, name in enumerate(self.channels)}
        self.groups = list(groups)
        self.accumulator = PairwiseAccumulator(len(self.channels), pairs_for_groups(self.groups))
        self.publisher = Publisher(self.accumulator, self.groups)

    @classmethod
    def from_handoff(cls, records) -> "Pipeline":
        """按交接记录（channels/groups）建管线。"""
        groups = [GroupSpec.from_record(g) for g in records["groups"]]
        return cls(records["channels"], groups)

    def submit(self, segment_id, rev: int, x, mask, w) -> SubmitOutcome:
        return self.accumulator.submit(segment_id, rev, x, mask, w)

    def publish(self) -> dict:
        return self.publisher.publish()

    def transform(self, reading) -> dict:
        """对一条现场读数按组部署白化。

        reading: 与通道对齐的序列（None 表示缺测）或 {通道名/通道号: 值}。
        返回 {"served": {组id: {version, channels, values}}, "skipped": {组id: 原因}}。
        """
        present = {}
        items = reading.items() if isinstance(reading, Mapping) else enumerate(reading)
        for key, value in items:
            if value is None:
                continue
            present[self._coerce_index(key)] = float(value)
        raw = self.publisher.transform(present)
        served = {
            gid: {
                "version": s["version"],
                "channels": [self.channels[c] for c in s["columns"]],
                "values": [float(v) for v in s["values"]],
            }
            for gid, s in raw["served"].items()
        }
        return {"served": served, "skipped": dict(raw["skipped"])}

    def report(self) -> dict:
        """审计报告：各配对样本量、各组采用版本与阻止原因（通道名口径）。

        训练端据此判断某批特征还能否混在一起消费：组 available/consumable
        为真且版本一致时才可共用该组的白化变换。
        """
        raw = self.publisher.report()
        groups = {}
        for gid, info in raw["groups"].items():
            info = dict(info)
            info["columns"] = [self.channels[c] for c in info["columns"]]
            info["consumable"] = info["available"]
            block = info.get("block")
            if block and block["reason"] == "coverage":
                block = {
                    "reason": block["reason"],
                    "detail": {
                        **block["detail"],
                        "pairs_without_estimate": [
                            [self.channels[i], self.channels[j]]
                            for i, j in block["detail"]["pairs_without_estimate"]
                        ],
                    },
                }
            info["block"] = block
            groups[gid] = info
        pairs = {}
        for gid, table in raw["pairs"].items():
            pairs[gid] = {
                f"{self.channels[i]}~{self.channels[j]}": value
                for (i, j), value in table.items()
            }
        return {"groups": groups, "pairs": pairs}

    def _coerce_index(self, key) -> int:
        if isinstance(key, str):
            return self._index[key]
        idx = int(key)
        if not 0 <= idx < len(self.channels):
            raise KeyError(f"通道号 {idx} 超出范围")
        return idx


def run_case(records, tensors) -> dict:
    """重放提交日志到部署：应用全部片段、发布、对交接读数做变换并出报告。

    submissions 按给定顺序应用；dataset 字段选择 tensors 中的
    ``<dataset>_x`` / ``<dataset>_mask`` / ``<dataset>_w``。
    """
    pipe = Pipeline.from_handoff(records)
    submissions = []
    for sub in records.get("submissions", []):
        dataset = sub["dataset"]
        outcome = pipe.submit(
            sub["id"],
            sub["rev"],
            tensors[f"{dataset}_x"],
            tensors[f"{dataset}_mask"],
            tensors[f"{dataset}_w"],
        )
        submissions.append(
            {
                "id": sub["id"],
                "rev": sub["rev"],
                "dataset": dataset,
                "outcome": outcome.value,
                "live_segments": pipe.accumulator.live_segments,
            }
        )
    publish = pipe.publish()
    result = {
        "submissions": submissions,
        "publish": {gid: d.to_dict() for gid, d in publish.items()},
        "report": pipe.report(),
    }
    reading = records.get("reading")
    if reading is not None:
        result["transform"] = pipe.transform(reading)
    return result
