import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up


def test_scan(tile_size=16, n_tiles=1):
    total_len = tile_size * tile_size * n_tiles
    device = "npu:0"
    dtype = torch.float32
    torch.npu.set_device(device)

    # Prepare Inputs
    # x = torch.ones(size=(total_len,), device="npu", dtype=dtype).contiguous()
    x = torch.linspace(1, total_len, steps=total_len, device="npu", dtype=dtype)
    s = torch.zeros_like(x)

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU JIT Kernel compilation
    file = "kernel_scan_v.cpp"
    scan_func = jit_compile(file)

    print(
        f"Testing NPU scan kernel: tile_size={tile_size}x{tile_size}, total_len={total_len} ({n_tiles} tiles)"
    )

    scan_func(x, s, total_len, tile_size)

    torch.npu.synchronize()

    print("Comparing results...")
    print("NPU scan result:\n", s.cpu()[:64])
    print("Input:\n", x.cpu()[:64])
    print("Expected:\n", expected_scan[:64])

    assert torch.allclose(
        s.cpu(), expected_scan, rtol=1e-3, atol=1e-3
    ), "Scan results do not match expected values!"

    print("All results matched. Scan test passed successfully.\n")

    clean_up(file)


if __name__ == "__main__":
    test_scan()
