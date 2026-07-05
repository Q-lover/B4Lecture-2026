#!/usr/bin/env python3
"""DiffusionModel Implementation Test.

実装した DiffusionModel の各メソッドをテストするスクリプト。

使用方法:
    python test_implementation.py
"""

import traceback

import diffusers
import torch
import torch.nn as nn
from diffusion_skeleton import DiffusionModel

# テスト共通パラメータ
T = 100
BATCH = 4
C, H, W = 3, 16, 16


def make_model():
    unet = diffusers.UNet2DModel(
        in_channels=C,
        out_channels=C,
        down_block_types=["DownBlock2D"],
        up_block_types=["UpBlock2D"],
        block_out_channels=[16],
    )
    return DiffusionModel(
        model=unet,
        criterion=nn.MSELoss(),
        num_timesteps=T,
        noise_schedule="linear",
        noise_schedule_kwargs={"start": 0.0001, "end": 0.02},
    ).eval()


def test_q_sample_shape():
    """q_sample の出力 shape テスト"""
    print("Testing DiffusionModel.q_sample (shape) ...")
    try:
        model = make_model()
        x0 = torch.randn(BATCH, C, H, W)
        t = torch.randint(0, T, (BATCH,))
        noise = torch.randn_like(x0)
        x_t = model.q_sample(x0, t, noise)
        assert x_t.shape == (BATCH, C, H, W), f"shape: {x_t.shape}"
        print("  ✅ q_sample shape: OK")
        return True
    except Exception as e:
        print(f"  ❌ q_sample shape: {e}")
        traceback.print_exc()
        return False


def test_q_sample_numerical():
    """q_sample の数値検証: t=0 のとき x_t = √ᾱ_0 · x_0（noise=0 で確認）"""
    print("Testing DiffusionModel.q_sample (numerical) ...")
    try:
        model = make_model()
        x0 = torch.ones(BATCH, C, H, W)
        t = torch.zeros(BATCH, dtype=torch.long)
        noise = torch.zeros_like(x0)
        x_t = model.q_sample(x0, t, noise)
        expected = model.alpha_prod[0].sqrt() * x0
        assert torch.allclose(
            x_t, expected, atol=1e-5
        ), f"期待値: {expected[0, 0, 0, 0]:.6f}  実際値: {x_t[0, 0, 0, 0]:.6f}"
        print("  ✅ q_sample numerical: OK")
        return True
    except Exception as e:
        print(f"  ❌ q_sample numerical: {e}")
        traceback.print_exc()
        return False


def test_q_sample_interpolation():
    """q_sample の数値検証: 任意の (x0, noise, t) で閉形式と一致するか"""
    print("Testing DiffusionModel.q_sample (formula check) ...")
    try:
        model = make_model()
        x0 = torch.randn(BATCH, C, H, W)
        t = torch.full((BATCH,), T // 2, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = model.q_sample(x0, t, noise)
        a = model.alpha_prod[T // 2].view(1, 1, 1, 1)
        expected = x0 * a.sqrt() + noise * (1 - a).sqrt()
        assert torch.allclose(x_t, expected, atol=1e-5), (
            f"q_sample の計算が正しくありません\n"
            f"  期待値: {expected[0, 0, 0, 0]:.6f}\n"
            f"  実際値: {x_t[0, 0, 0, 0]:.6f}"
        )
        print("  ✅ q_sample formula: OK")
        return True
    except Exception as e:
        print(f"  ❌ q_sample formula: {e}")
        traceback.print_exc()
        return False


def test_p_sample_shape():
    """p_sample の出力 shape テスト"""
    print("Testing DiffusionModel.p_sample (shape) ...")
    try:
        model = make_model()
        x = torch.randn(BATCH, C, H, W)
        t = torch.zeros(BATCH, dtype=torch.long)
        x_prev = model.p_sample(x, t)
        assert x_prev.shape == (BATCH, C, H, W), f"shape: {x_prev.shape}"
        print("  ✅ p_sample shape: OK")
        return True
    except Exception as e:
        print(f"  ❌ p_sample shape: {e}")
        traceback.print_exc()
        return False


def test_p_sample_t0_deterministic():
    """t=0 のとき p_sample は確定論的（ノイズを加えない）"""
    print("Testing DiffusionModel.p_sample (t=0 is deterministic) ...")
    try:
        model = make_model()
        x = torch.randn(BATCH, C, H, W)
        t = torch.zeros(BATCH, dtype=torch.long)
        x_prev1 = model.p_sample(x, t)
        x_prev2 = model.p_sample(x, t)
        assert torch.allclose(
            x_prev1, x_prev2
        ), "t=0 のとき p_sample は確定論的であるべきです（ノイズを加えてはいけない）"
        print("  ✅ p_sample (t=0 deterministic): OK")
        return True
    except Exception as e:
        print(f"  ❌ p_sample (t=0 deterministic): {e}")
        traceback.print_exc()
        return False


def test_p_sample_t_nonzero_stochastic():
    """t>0 のとき p_sample は確率的（ノイズを加える）"""
    print("Testing DiffusionModel.p_sample (t>0 is stochastic) ...")
    try:
        model = make_model()
        x = torch.randn(BATCH, C, H, W)
        t = torch.full((BATCH,), T // 2, dtype=torch.long)
        x_prev1 = model.p_sample(x, t)
        x_prev2 = model.p_sample(x, t)
        assert not torch.allclose(
            x_prev1, x_prev2
        ), "t>0 のとき p_sample は確率的であるべきです（ノイズを加える必要がある）"
        print("  ✅ p_sample (t>0 stochastic): OK")
        return True
    except Exception as e:
        print(f"  ❌ p_sample (t>0 stochastic): {e}")
        traceback.print_exc()
        return False


def test_training_step_scalar():
    """training_step がスカラー損失（≥ 0）を返すテスト"""
    print("Testing DiffusionModel.training_step (scalar loss) ...")
    try:
        model = make_model().train()
        images = torch.randn(BATCH, C, H, W)
        loss = model.training_step(images)
        assert loss.dim() == 0, f"loss はスカラーであるべき: dim={loss.dim()}"
        assert loss.item() >= 0, f"MSE 損失は非負であるべき: {loss.item():.4f}"
        print("  ✅ training_step scalar loss: OK")
        return True
    except Exception as e:
        print(f"  ❌ training_step scalar loss: {e}")
        traceback.print_exc()
        return False


def test_training_step_gradient():
    """loss.backward() で全パラメータに勾配が流れるかのテスト"""
    print("Testing DiffusionModel.training_step (gradient flow) ...")
    try:
        model = make_model().train()
        images = torch.randn(BATCH, C, H, W)
        loss = model.training_step(images)
        loss.backward()
        has_grad = all(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "一部のパラメータに勾配が流れていません"
        print("  ✅ training_step gradient flow: OK")
        return True
    except Exception as e:
        print(f"  ❌ training_step gradient flow: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 55)
    print("Ex8 DiffusionModel - Implementation Test")
    print("=" * 55)

    tests = [
        test_q_sample_shape,
        test_q_sample_numerical,
        test_q_sample_interpolation,
        test_p_sample_shape,
        test_p_sample_t0_deterministic,
        test_p_sample_t_nonzero_stochastic,
        test_training_step_scalar,
        test_training_step_gradient,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ テスト実行エラー: {e}")
            results.append(False)
        print()

    print("=" * 55)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("実装完了です。main.py で学習を開始できます。")
        print("\n推奨コマンド:")
        print("  python main.py datadir=<データダウンロード先のパス>")
    else:
        print(f"⚠️  {total - passed} tests failed. ({passed}/{total})")
        print("diffusion_skeleton.py の #TODO 箇所を確認してください。")


if __name__ == "__main__":
    main()
