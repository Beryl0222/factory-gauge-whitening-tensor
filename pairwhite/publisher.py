"""按组发布白化变换：覆盖判定、双限制合规、版本保持。

- 组覆盖 = 组内所有通道对（含对角线）有效样本量 (sum w)^2/sum(w^2) 的最小值，
  按大于等于判定。无共测的通道对没有估计，不允许填零冒充验证完成。
- 覆盖达标后评估对称白化：谱范数上限与修正量上限（Frobenius）须同时合规；
  无法同守时该组保留上一次有效变换，没有上一份就标为不可用，其他组继续更新。
- 单通道中心值与该组白化矩阵同版交付；组被阻止换版时均值也保持旧版。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accumulator import PairwiseAccumulator
from .common import TOL
from .whitening import symmetric_whitening


@dataclass(frozen=True)
class GroupSpec:
    """交接资料中可独立继续的通道组及其发布限制。"""

    gid: str
    columns: tuple
    coverage: float         # 覆盖下限：组内最小配对有效样本量
    max_norm: float         # 噪声放大上限：交付变换的谱范数上限
    max_adjust_fro: float   # 最大可接受的协方差修正量（Frobenius）

    @staticmethod
    def from_record(rec) -> "GroupSpec":
        return GroupSpec(
            gid=str(rec["id"]),
            columns=tuple(int(c) for c in rec["columns"]),
            coverage=float(rec["coverage"]),
            max_norm=float(rec["max_norm"]),
            max_adjust_fro=float(rec["max_adjust_fro"]),
        )


def group_pairs(columns) -> list:
    """组内全部通道对（含对角线），按列顺序枚举上三角。"""
    cols = list(columns)
    return [(cols[a], cols[b]) for a in range(len(cols)) for b in range(a, len(cols))]


def pairs_for_groups(groups) -> list:
    """所有分组声明的通道对并集（规范化 i<=j）。"""
    out = set()
    for g in groups:
        for i, j in group_pairs(g.columns):
            out.add((min(i, j), max(i, j)))
    return sorted(out)


@dataclass
class PublishedTransform:
    """一组已发布的同版变换：白化矩阵 + 单通道中心值。"""

    version: int
    whitening: np.ndarray  # (k, k) 对称
    means: np.ndarray      # (k,) 与 columns 对齐
    coverage: float
    adjust_fro: float
    spectral_norm: float


@dataclass
class PublishDecision:
    """一次发布评估中单个组的结果。"""

    gid: str
    decision: str           # published / held / unavailable
    version: int | None     # 当前生效版本；从未发布过为 None
    coverage: float
    coverage_required: float
    adjust_fro: float | None
    block: dict | None      # 阻止更新的原因；未阻止为 None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "version": self.version,
            "coverage": self.coverage,
            "coverage_required": self.coverage_required,
            "adjust_fro": self.adjust_fro,
            "block": self.block,
        }


class Publisher:
    """读取累积器统计，按组做发布决策并维护已发布版本。"""

    def __init__(self, accumulator: PairwiseAccumulator, groups):
        self.acc = accumulator
        self.groups = list(groups)
        seen = set()
        for g in self.groups:
            if g.gid in seen:
                raise ValueError(f"组 id 重复: {g.gid!r}")
            seen.add(g.gid)
            if not g.columns or len(set(g.columns)) != len(g.columns):
                raise ValueError(f"组 {g.gid!r} 的通道列无效")
            for c in g.columns:
                if not 0 <= c < accumulator.n_channels:
                    raise ValueError(f"组 {g.gid!r} 的通道 {c} 超出范围")
            for i, j in group_pairs(g.columns):
                accumulator.pair_summary(i, j)  # 未声明的通道对在此抛 KeyError
            if g.coverage < 0:
                raise ValueError(f"组 {g.gid!r} 的覆盖下限不能为负")
        self._published: dict[str, PublishedTransform | None] = {g.gid: None for g in self.groups}
        self._versions = {g.gid: 0 for g in self.groups}
        self._last_block = {g.gid: None for g in self.groups}
        self._last_eval = {g.gid: None for g in self.groups}

    def group_coverage(self, g: GroupSpec) -> tuple[float, list]:
        """组覆盖与无估计通道对清单。覆盖含对角线，取组内最小配对有效量。"""
        best = float("inf")
        missing = []
        for i, j in group_pairs(g.columns):
            s = self.acc.pair_summary(i, j)
            if not s.has_estimate:
                missing.append([i, j])
            best = min(best, s.n_eff)
        return float(best), missing

    def publish(self) -> dict:
        """对全部组做一次发布评估；组间互不影响。"""
        return {g.gid: self._publish_group(g) for g in self.groups}

    def _publish_group(self, g: GroupSpec) -> PublishDecision:
        coverage, missing = self.group_coverage(g)
        held = self._versions[g.gid] or None
        if missing or coverage + TOL < g.coverage:
            block = {
                "reason": "coverage",
                "detail": {
                    "coverage": coverage,
                    "required": g.coverage,
                    "pairs_without_estimate": missing,
                },
            }
            self._last_block[g.gid] = block
            self._last_eval[g.gid] = {"coverage": coverage, "adjust_fro": None}
            return PublishDecision(
                g.gid, "held" if held else "unavailable",
                held, coverage, g.coverage, None, block,
            )
        k = len(g.columns)
        cov = np.empty((k, k))
        for a, i in enumerate(g.columns):
            for b, j in enumerate(g.columns):
                cov[a, b] = self.acc.pair_summary(i, j).cov
        outcome = symmetric_whitening(cov, g.max_norm, g.max_adjust_fro)
        self._last_eval[g.gid] = {"coverage": coverage, "adjust_fro": outcome.adjust_fro}
        if not outcome.compliant:
            block = {
                "reason": "limits",
                "detail": {
                    "adjust_fro_required": outcome.adjust_fro,
                    "max_adjust_fro": g.max_adjust_fro,
                    "min_eigenvalue": outcome.min_eigenvalue_raw,
                    "eigen_floor": outcome.eigen_floor,
                    "max_norm": g.max_norm,
                },
            }
            self._last_block[g.gid] = block
            return PublishDecision(
                g.gid, "held" if held else "unavailable",
                held, coverage, g.coverage, outcome.adjust_fro, block,
            )
        version = self._versions[g.gid] + 1
        self._versions[g.gid] = version
        means = np.array([self.acc.channel_summary(c).mean for c in g.columns])
        self._published[g.gid] = PublishedTransform(
            version=version,
            whitening=outcome.whitening,
            means=means,
            coverage=coverage,
            adjust_fro=outcome.adjust_fro,
            spectral_norm=outcome.spectral_norm,
        )
        self._last_block[g.gid] = None
        return PublishDecision(
            g.gid, "published", version, coverage, g.coverage, outcome.adjust_fro, None,
        )

    def published(self, gid: str) -> PublishedTransform | None:
        return self._published[gid]

    def transform(self, present: dict) -> dict:
        """对一条现场读数按组部署白化。

        present: {通道号: 读数值}。只有组内通道齐全且组已发布时才返回结果；
        每组使用同一份版本的白化矩阵与均值，不混用不同新鲜度的版本。
        """
        served = {}
        skipped = {}
        for g in self.groups:
            pub = self._published[g.gid]
            if pub is None:
                skipped[g.gid] = "unavailable: 组没有已发布的有效变换"
                continue
            missing = [c for c in g.columns if c not in present]
            if missing:
                skipped[g.gid] = f"incomplete: 读数缺少通道 {missing}"
                continue
            x = np.array([present[c] for c in g.columns], dtype=np.float64)
            y = pub.whitening @ (x - pub.means)
            served[g.gid] = {
                "version": pub.version,
                "columns": list(g.columns),
                "values": y,
            }
        return {"served": served, "skipped": skipped}

    def report(self) -> dict:
        """审计信息：各组版本与阻止原因、各配对样本量（索引口径）。"""
        groups = {}
        for g in self.groups:
            pub = self._published[g.gid]
            coverage, _ = self.group_coverage(g)
            ev = self._last_eval[g.gid] or {}
            groups[g.gid] = {
                "columns": list(g.columns),
                "version": pub.version if pub else None,
                "available": pub is not None,
                "coverage": float(coverage),
                "coverage_required": g.coverage,
                "adjust_fro": ev.get("adjust_fro"),
                "spectral_norm": pub.spectral_norm if pub else None,
                "block": self._last_block[g.gid],
            }
        pairs = {}
        for g in self.groups:
            table = {}
            for i, j in group_pairs(g.columns):
                s = self.acc.pair_summary(i, j)
                table[(i, j)] = None if not s.has_estimate else {
                    "n_eff": s.n_eff,
                    "n": s.n,
                    "sum_w": s.sum_w,
                }
            pairs[g.gid] = table
        return {"groups": groups, "pairs": pairs}
