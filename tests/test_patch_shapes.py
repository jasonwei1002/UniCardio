"""Patch tokenization 重构后的形状契约测试。

验证：
1. forward 外部契约保持 ``(B, 1, 3*L_slot) → (B, 1, L_slot)``，对多种 patch_size 都成立；
2. 内部 transformer 序列长度按 patch 粒度缩减；
3. sinusoidal PE buffer 长度按 patch 粒度生成；
4. ``slot_length % patch_size != 0`` 时构造失败；
5. ``n_patches_per_slot`` 在 model / backbone 上可访问。

实现详见 reports/vit-patch-model-reactive-barto.md。
"""

from __future__ import annotations

import pytest
import torch

from src.model_module.attention_masks import build_task_mask, clear_mask_cache
from src.model_module.backbone import BackboneConfig
from src.model_module.tasks import TASK_LIST
from src.model_module.unicardio_rf import UniCardioRF

# 单元测试用最小的可工作 slot_length。50 能被 5/10/25/50 整除，方便 patch 网格扫描。
SLOT_LEN = 50
B = 2


def _make_model(patch_size: int, slot_length: int = SLOT_LEN) -> UniCardioRF:
    torch.manual_seed(0)
    cfg = BackboneConfig(
        slot_length=slot_length,
        channels=288,
        n_layers=2,
        nheads=4,
        time_embedding_dim=64,
        ffn_dim=32,
        patch_size=patch_size,
    )
    return UniCardioRF(cfg)


@pytest.fixture(autouse=True)
def _reset_mask_cache():
    clear_mask_cache()
    yield
    clear_mask_cache()


@pytest.mark.parametrize("patch_size", [5, 10, 25, 50])
def test_forward_shape_invariant(patch_size: int) -> None:
    """forward 输入 ``(B, 1, 3*L_slot)``，输出始终 ``(B, 1, L_slot)``，无论 patch_size。"""
    model = _make_model(patch_size)
    x_full = torch.randn(B, 1, 3 * SLOT_LEN)
    t = torch.rand(B)
    n_patches = SLOT_LEN // patch_size
    mask = build_task_mask(TASK_LIST[0].name, n_patches, device="cpu", dtype=torch.bool)
    out = model(x_full, t, mask, target_slot=int(TASK_LIST[0].target_slot))
    assert out.shape == (B, 1, SLOT_LEN)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("patch_size", [5, 10, 25, 50])
def test_pe_buffer_size_matches_patch_resolution(patch_size: int) -> None:
    """sinusoidal PE 在 patch 粒度上注入；buffer 长度 = ``3 * n_patches_per_slot``。"""
    model = _make_model(patch_size)
    n_patches = SLOT_LEN // patch_size
    assert model.backbone.pe.shape == (288, 3 * n_patches)


@pytest.mark.parametrize("n_patches_per_slot", [2, 5, 10, 50])
def test_mask_shape_at_patch_resolution(n_patches_per_slot: int) -> None:
    """``build_task_mask`` 调用方传入 patch 数（而非 sample 数），mask 是 ``(3N, 3N)``。"""
    mask = build_task_mask("ecg2abp", n_patches_per_slot, device="cpu", dtype=torch.bool)
    assert mask.shape == (3 * n_patches_per_slot, 3 * n_patches_per_slot)


def test_patch_size_must_divide_slot_length() -> None:
    """``slot_length % patch_size != 0`` 必须在构造时抛 ValueError，避免 ConvTranspose 长度对不齐。"""
    with pytest.raises(ValueError, match=r"slot_length.*patch_size"):
        BackboneConfig(
            slot_length=50,
            channels=288,
            n_layers=2,
            nheads=4,
            time_embedding_dim=64,
            ffn_dim=32,
            patch_size=24,
        )


def test_n_patches_per_slot_exposed_on_model_and_backbone() -> None:
    """``rf_train_step`` / ``euler_sample`` 通过 ``model.n_patches_per_slot`` 拿 mask 长度。"""
    model = _make_model(patch_size=10)
    assert model.n_patches_per_slot == 5
    assert model.backbone.n_patches_per_slot == 5
