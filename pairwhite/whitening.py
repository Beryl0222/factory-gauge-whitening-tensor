"""对称白化与双限制合规判定。

谱范数限制 ||W||_2 <= max_norm 等价于所交付修正协方差的最小特征值不低于
1 / max_norm^2（W 为对称逆平方根）。把特征值截断到该下限是满足谱约束的
矩阵中 Frobenius 意义下离原始协方差最近的修正；因此若截断修正量仍超出
max_adjust_fro，则不存在同时满足两个限制的对称白化，该组应保留旧版。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import TOL


@dataclass
class WhiteningOutcome:
    """一次白化评估的结果。"""

    compliant: bool
    covariance_raw: np.ndarray        # 对称化后的原始配对协方差
    covariance_adjusted: np.ndarray   # 谱截断后的最近可行修正（无论是否合规）
    whitening: np.ndarray | None      # 合规时交付的对称逆平方根，否则 None
    adjust_fro: float                 # ||修正量||_F
    spectral_norm: float | None       # 交付变换的欧氏算子范数
    eigen_floor: float                # 1 / max_norm^2
    min_eigenvalue_raw: float


def symmetric_whitening(covariance, max_norm: float, max_adjust_fro: float) -> WhiteningOutcome:
    """对配对协方差做最近可行修正并给出对称白化。

    修正量按协方差矩阵的 Frobenius 范数计量；两个限制（谱范数上限、
    修正量上限）无法同守时 compliant=False，调用方保留上一份有效变换。
    """
    if max_norm <= 0:
        raise ValueError("谱范数上限必须为正")
    if max_adjust_fro < 0:
        raise ValueError("修正量上限不能为负")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("协方差矩阵必须为方阵")
    sym = (cov + cov.T) * 0.5
    evals, evecs = np.linalg.eigh(sym)
    floor = 1.0 / (max_norm * max_norm)
    adjusted = np.maximum(evals, floor)
    delta = adjusted - evals
    adjust_fro = float(np.sqrt(np.dot(delta, delta)))
    c_adj = (evecs * adjusted) @ evecs.T
    c_adj = (c_adj + c_adj.T) * 0.5
    compliant = adjust_fro <= max_adjust_fro + TOL
    whitening = None
    spectral = None
    if compliant:
        inv = 1.0 / np.sqrt(adjusted)
        w = (evecs * inv) @ evecs.T
        whitening = (w + w.T) * 0.5
        spectral = float(1.0 / np.sqrt(adjusted[0]))
    return WhiteningOutcome(
        compliant=compliant,
        covariance_raw=sym,
        covariance_adjusted=c_adj,
        whitening=whitening,
        adjust_fro=adjust_fro,
        spectral_norm=spectral,
        eigen_floor=float(floor),
        min_eigenvalue_raw=float(evals[0]),
    )
