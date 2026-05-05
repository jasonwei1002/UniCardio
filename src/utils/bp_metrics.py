"""SBP / DBP 提取与误差统计。

通过 pyvital 的经典 DSP 峰检测器（``detect_peaks``）在每个 ABP 窗口里定位逐搏
收缩峰（max）与舒张谷（min），然后对一窗内的多搏取均值得到 mean SBP / mean DBP。
基于此给出 batch 级别的 MAE / ME / std，可直接对照 AAMI / BHS 标准。

输入信号必须是反归一化后的 mmHg 值（不是 ``(x-100)/50`` 后的模型空间）。
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvital  # noqa: F401
    _HAS_PYVITAL = True
except ImportError:  # pragma: no cover
    _HAS_PYVITAL = False
    logger.warning(
        "pyvital not installed; SBP/DBP metrics will be skipped. "
        "Install via `pip install pyvital`."
    )


def pyvital_available() -> bool:
    """Public capability flag: True iff pyvital import succeeded.

    Use this instead of importing the private ``_HAS_PYVITAL`` symbol from
    other modules — keeps the dependency check decoupled from the detection
    mechanism.
    """
    return _HAS_PYVITAL


# 生理合理范围（mmHg）。命中外面的搏拍直接丢弃，避免野点污染均值。
_SBP_MIN, _SBP_MAX = 60.0, 220.0
_DBP_MIN, _DBP_MAX = 30.0, 130.0
_PP_MIN, _PP_MAX = 20.0, 100.0


def _silent_detect_peaks(
    abp_mmHg: np.ndarray, srate: int
) -> tuple[list[int], list[int]]:
    """屏蔽 pyvital 的诊断 stdout（如 "HR estimation failed, assume 75"），避免日志被刷屏。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return pyvital.detect_peaks(abp_mmHg, srate)


def extract_sbp_dbp(
    abp_mmHg: np.ndarray, srate: int
) -> Optional[tuple[float, float]]:
    """从单个 1D ABP 窗口（mmHg）提取 mean SBP / mean DBP。

    Args:
        abp_mmHg: 1D numpy 数组，mmHg 物理量纲。
        srate: 采样率（Hz）。pyvital 内部会重采样到 100 Hz 跑算法，
            最终峰位再回到 raw_srate 上精修，所以直接传你的原始采样率即可。

    Returns:
        ``(mean_sbp, mean_dbp)`` 或 ``None``（pyvital 缺失 / 信号异常 /
        无任何合规搏拍）。
    """
    if not _HAS_PYVITAL:
        return None

    arr = np.asarray(abp_mmHg)
    if arr.ndim > 1:
        arr = arr.squeeze()
    if arr.ndim != 1:
        return None
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        return None

    try:
        minlist, maxlist = _silent_detect_peaks(arr, int(srate))
    except Exception as e:  # noqa: BLE001
        logger.debug("detect_peaks failed: %s", e)
        return None

    if len(maxlist) < 2 or len(minlist) < 1:
        return None

    sbps: list[float] = []
    dbps: list[float] = []
    # pyvital: minlist[i] 落在 maxlist[i] 与 maxlist[i+1] 之间，所以最多
    # 配 min(len(minlist), len(maxlist)-1) 对。
    n_pairs = min(len(minlist), len(maxlist) - 1)
    for i in range(n_pairs):
        sbp = float(arr[maxlist[i]])
        dbp = float(arr[minlist[i]])
        pp = sbp - dbp
        if not (_PP_MIN < pp < _PP_MAX):
            continue
        if not (_SBP_MIN < sbp < _SBP_MAX):
            continue
        if not (_DBP_MIN < dbp < _DBP_MAX):
            continue
        sbps.append(sbp)
        dbps.append(dbp)

    if not sbps:
        return None
    return float(np.mean(sbps)), float(np.mean(dbps))


def bp_per_sample_errors(
    pred_mmHg: np.ndarray,
    target_mmHg: np.ndarray,
    srate: int,
) -> tuple[list[float], list[float], int]:
    """逐 sample 提取 (SBP, DBP) 并算 ``pred − target`` 误差，返回原始数组。

    供跨 batch 累加用：``bp_errors`` 是一次性版本，在累加场景下不能直接合并
    （MAE / std 不是线性聚合）。本函数把"提取 + 配对"的部分单独拿出来，让
    上层累完所有 batch 后再统一聚合。

    Args:
        pred_mmHg / target_mmHg: 形状 ``(B, 1, L)`` 或 ``(B, L)``。
        srate: 采样率（Hz）。

    Returns:
        ``(sbp_errs, dbp_errs, n_total)``——pyvital 缺失或某窗提取失败的样本
        会从误差列表里被丢掉，但仍计入 ``n_total``。
    """
    if not _HAS_PYVITAL:
        n_total = int(np.asarray(pred_mmHg).shape[0])
        return [], [], n_total

    p = np.asarray(pred_mmHg)
    t = np.asarray(target_mmHg)
    if p.ndim == 3:
        p = p[:, 0, :]
    if t.ndim == 3:
        t = t[:, 0, :]
    if p.shape != t.shape:
        raise ValueError(f"pred shape {p.shape} != target shape {t.shape}")

    sbp_errs: list[float] = []
    dbp_errs: list[float] = []
    for pi, ti in zip(p, t):
        rp = extract_sbp_dbp(pi, srate)
        rt = extract_sbp_dbp(ti, srate)
        if rp is None or rt is None:
            continue
        sbp_errs.append(rp[0] - rt[0])
        dbp_errs.append(rp[1] - rt[1])
    return sbp_errs, dbp_errs, int(p.shape[0])


def bp_aggregate_errors(
    sbp_errs: list[float],
    dbp_errs: list[float],
    n_total: int,
) -> dict[str, float]:
    """把 :func:`bp_per_sample_errors` 累出来的原始误差列表聚合成 MAE / ME / std。

    pyvital 缺失（``_HAS_PYVITAL=False``）时返回空字典；列表非空但 0 valid
    （所有 sample 都被丢弃）时返回带 NaN 的字典，``n_valid=0``。
    """
    if not _HAS_PYVITAL:
        return {}
    n_valid = len(sbp_errs)
    if n_valid == 0:
        return {
            "sbp_mae": float("nan"),
            "dbp_mae": float("nan"),
            "sbp_me": float("nan"),
            "dbp_me": float("nan"),
            "sbp_std": float("nan"),
            "dbp_std": float("nan"),
            "n_valid": 0,
            "n_total": n_total,
        }
    s = np.asarray(sbp_errs, dtype=np.float64)
    d = np.asarray(dbp_errs, dtype=np.float64)
    return {
        "sbp_mae": float(np.abs(s).mean()),
        "dbp_mae": float(np.abs(d).mean()),
        "sbp_me": float(s.mean()),
        "dbp_me": float(d.mean()),
        "sbp_std": float(s.std(ddof=0)),
        "dbp_std": float(d.std(ddof=0)),
        "n_valid": n_valid,
        "n_total": n_total,
    }


def bp_errors(
    pred_mmHg: np.ndarray,
    target_mmHg: np.ndarray,
    srate: int,
) -> dict[str, float]:
    """对一批 ABP 重建结果计算 SBP / DBP 误差。

    Args:
        pred_mmHg: 形状 ``(B, 1, L)`` 或 ``(B, L)``，预测 ABP（mmHg）。
        target_mmHg: 与 ``pred_mmHg`` 同形状，参考 ABP（mmHg）。
        srate: 采样率（Hz）。

    Returns:
        含 ``sbp_mae / dbp_mae / sbp_me / dbp_me / sbp_std / dbp_std /
        n_valid / n_total`` 的字典；pyvital 不可用时返回空字典。
        所有误差均按「pred − target」给出（正值 = 系统性高估）。
    """
    sbp_errs, dbp_errs, n_total = bp_per_sample_errors(pred_mmHg, target_mmHg, srate)
    return bp_aggregate_errors(sbp_errs, dbp_errs, n_total)
