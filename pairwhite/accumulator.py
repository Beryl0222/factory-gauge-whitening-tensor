"""成对加权充分统计量的流式累积器。

设计要点：

- 只跟踪交接分组声明的通道对（含对角线）。状态大小为
  O(通道数 + 组内通道对数 + 存活片段贡献)，不随样本数平方增长，
  也从不物化完整特征矩阵。
- 行分块处理，瞬态缓冲远低于交接的 1024 行 / 2MB 暂存上限。
- 每个已应用片段保留一份稀疏贡献与内容摘要：内容一致的重发不重复计入；
  更正片段（同 id、不同内容）先撤销旧贡献再计入新贡献，新旧有效通道对
  的统计随之同步更新。
- 权重为零的样本没有任何贡献；缺失通道由有效位表达。
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum

import numpy as np

_ROWS_PER_CHUNK = 256  # 单块行数，低于交接的 1024 行上限
_PAIR_BLOCK = 64       # 单次向量化的通道对数，控制瞬态缓冲

_CHANNEL_FIELDS = 4  # sum_w, sum_wx, sum_w2, n
_PAIR_FIELDS = 6     # sum_w, sum_wx, sum_wy, sum_wxy, sum_w2, n


class SubmitOutcome(Enum):
    """片段提交结果。"""

    APPLIED = "applied"        # 新片段，已计入
    DUPLICATE = "duplicate"    # 内容一致的重发，未重复计入
    REPLACED = "replaced"      # 更正片段：已撤销旧贡献并计入新贡献


@dataclass
class ChannelSummary:
    """单通道汇总（该通道全部有效正权样本）。"""

    sum_w: float
    n: int
    n_eff: float   # (sum w)^2 / sum(w^2)
    mean: float    # 无有效样本时为 nan


@dataclass
class PairSummary:
    """通道对汇总（只看两侧共有效的正权样本）。"""

    has_estimate: bool  # 是否存在正权共测样本；无共测的通道对没有估计
    sum_w: float
    n: int
    n_eff: float        # (sum w)^2 / sum(w^2)
    mean_i: float       # 配对均值，不代入单通道均值
    mean_j: float
    cov: float          # 配对归一化后的加权总体协方差


@dataclass
class _Contribution:
    """单个片段对累积器的稀疏贡献，用于撤销与去重。"""

    rev: int
    digest: bytes
    channel_idx: np.ndarray  # (k,) int64
    channel_vals: np.ndarray  # (k, _CHANNEL_FIELDS) float64
    pair_idx: np.ndarray     # (m,) int64
    pair_vals: np.ndarray    # (m, _PAIR_FIELDS) float64


def _shape2(a) -> tuple[int, int]:
    shape = getattr(a, "shape", None)
    if shape is not None:
        return int(shape[0]), int(shape[1])
    return len(a), len(a[0])


class PairwiseAccumulator:
    """按声明通道对维护加权充分统计量的累积器。"""

    def __init__(self, n_channels: int, pairs):
        if n_channels <= 0:
            raise ValueError("通道数必须为正")
        normalized = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs})
        for i, j in normalized:
            if not (0 <= i < n_channels and 0 <= j < n_channels):
                raise ValueError(f"通道对 {(i, j)} 超出通道范围")
        self.n_channels = n_channels
        self.pairs = normalized
        self._pair_i = np.array([p[0] for p in normalized], dtype=np.int64)
        self._pair_j = np.array([p[1] for p in normalized], dtype=np.int64)
        self._lookup = np.full((n_channels, n_channels), -1, dtype=np.int64)
        for k, (i, j) in enumerate(normalized):
            self._lookup[i, j] = k
            self._lookup[j, i] = k
        self.channel_stats = np.zeros((n_channels, _CHANNEL_FIELDS))
        self.pair_stats = np.zeros((len(normalized), _PAIR_FIELDS))
        self._segments: dict[object, _Contribution] = {}

    @property
    def live_segments(self) -> int:
        """当前存活（已计入）的片段数。"""
        return len(self._segments)

    def submit(self, segment_id, rev: int, x, mask, w) -> SubmitOutcome:
        """提交一个特征片段。

        同一 segment_id 内容一致的重发不重复计入；同 id 不同内容视为更正，
        先撤销旧贡献再计入新贡献。同 id 同 rev 但内容冲突时报错。
        """
        rows, cols = _shape2(x)
        if rows == 0:
            raise ValueError("空片段")
        if cols != self.n_channels:
            raise ValueError(f"片段列数 {cols} 与通道数 {self.n_channels} 不一致")
        if _shape2(mask) != (rows, cols):
            raise ValueError("有效位形状与观测不一致")
        if len(w) != rows:
            raise ValueError("权重长度与行数不一致")
        contrib = self._contribution(rev, x, mask, w, rows)
        old = self._segments.get(segment_id)
        if old is not None and old.digest == contrib.digest:
            old.rev = rev  # 内容一致的重发：只更新修订号，不重复计入
            return SubmitOutcome.DUPLICATE
        if old is not None and old.rev == rev:
            raise ValueError(f"片段 {segment_id!r} 修订 {rev} 与已收内容冲突")
        if old is not None:
            self._retract(old)
        self._apply(contrib)
        self._segments[segment_id] = contrib
        return SubmitOutcome.REPLACED if old is not None else SubmitOutcome.APPLIED

    def channel_summary(self, c: int) -> ChannelSummary:
        sw, sx, sw2, n = self.channel_stats[c]
        n_eff = float(sw * sw / sw2) if sw2 > 0 else 0.0
        mean = float(sx / sw) if sw > 0 else float("nan")
        return ChannelSummary(float(sw), int(n), n_eff, mean)

    def pair_summary(self, i: int, j: int) -> PairSummary:
        k = int(self._lookup[i, j])
        if k < 0:
            raise KeyError(f"通道对 {(i, j)} 未在交接分组中声明")
        sw, sx, sy, sxy, sw2, n = (float(v) for v in self.pair_stats[k])
        if self.pairs[k][0] != i:  # 调用方向与存储方向相反，交换两侧
            sx, sy = sy, sx
        if sw <= 0:
            return PairSummary(False, 0.0, int(n), 0.0, float("nan"), float("nan"), float("nan"))
        mean_i = sx / sw
        mean_j = sy / sw
        cov = sxy / sw - mean_i * mean_j  # 加权总体口径，配对归一化
        n_eff = float(sw * sw / sw2) if sw2 > 0 else 0.0
        return PairSummary(True, sw, int(n), n_eff, mean_i, mean_j, cov)

    def _apply(self, contrib: _Contribution) -> None:
        self.channel_stats[contrib.channel_idx] += contrib.channel_vals
        self.pair_stats[contrib.pair_idx] += contrib.pair_vals

    def _retract(self, contrib: _Contribution) -> None:
        self.channel_stats[contrib.channel_idx] -= contrib.channel_vals
        self.pair_stats[contrib.pair_idx] -= contrib.pair_vals

    def _contribution(self, rev: int, x, mask, w, rows: int) -> _Contribution:
        ch = np.zeros((self.n_channels, _CHANNEL_FIELDS))
        pa = np.zeros((len(self.pairs), _PAIR_FIELDS))
        digest = hashlib.sha256()
        digest.update(struct.pack("<QQ", rows, self.n_channels))
        for start in range(0, rows, _ROWS_PER_CHUNK):
            stop = min(start + _ROWS_PER_CHUNK, rows)
            xc = np.asarray(x[start:stop], dtype=np.float64)
            mc = np.asarray(mask[start:stop], dtype=bool)
            wc = np.asarray(w[start:stop], dtype=np.float64).reshape(-1)
            if wc.shape[0] != stop - start:
                raise ValueError("权重长度与行数不一致")
            if not (np.all(np.isfinite(wc)) and np.all(wc >= 0)):
                raise ValueError("权重必须为有限非负数")
            if not np.all(np.isfinite(xc[mc])):
                raise ValueError("有效位置上的观测必须为有限数")
            digest.update(np.ascontiguousarray(xc, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(mc, dtype=np.uint8).tobytes())
            digest.update(np.ascontiguousarray(wc, dtype="<f8").tobytes())
            wpos = wc > 0
            if not np.any(wpos):
                continue  # 零权样本无贡献
            wv = np.where(wpos, wc, 0.0)
            wm = mc * wv[:, None]           # 无效或零权处为 0
            xv = np.where(mc, xc, 0.0)
            ch[:, 0] += wm.sum(axis=0)
            ch[:, 1] += (wm * xv).sum(axis=0)
            ch[:, 2] += (wm * wv[:, None]).sum(axis=0)
            ch[:, 3] += (wm > 0).sum(axis=0)
            for p0 in range(0, len(self.pairs), _PAIR_BLOCK):
                sl = slice(p0, min(p0 + _PAIR_BLOCK, len(self.pairs)))
                ii = self._pair_i[sl]
                jj = self._pair_j[sl]
                co = mc[:, ii] & mc[:, jj]
                co &= wpos[:, None]
                if not co.any():
                    continue
                ww = co * wv[:, None]
                xi = np.where(co, xc[:, ii], 0.0)
                xj = np.where(co, xc[:, jj], 0.0)
                pa[sl, 0] += ww.sum(axis=0)
                pa[sl, 1] += (ww * xi).sum(axis=0)
                pa[sl, 2] += (ww * xj).sum(axis=0)
                pa[sl, 3] += (ww * xi * xj).sum(axis=0)
                pa[sl, 4] += (ww * wv[:, None]).sum(axis=0)
                pa[sl, 5] += co.sum(axis=0)
        cidx = np.flatnonzero(ch[:, 0] != 0.0)
        pidx = np.flatnonzero(pa[:, 0] != 0.0)
        return _Contribution(
            rev=rev,
            digest=digest.digest(),
            channel_idx=cidx,
            channel_vals=ch[cidx].copy(),
            pair_idx=pidx,
            pair_vals=pa[pidx].copy(),
        )
