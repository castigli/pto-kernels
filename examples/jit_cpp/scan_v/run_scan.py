import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up


def test_scan(tile_size=16, n_tiles=1):
    total_len = tile_size * tile_size * n_tiles
    device = "npu:0"
    dtype = torch.float32
    torch.npu.set_device(device)

    shift_chunk = torch.zeros(size=(total_len,), device=device, dtype=torch.int32)
    shift_chunk[0] = 7
    shift_chunk[1] = 0
    shift_chunk[2] = 1
    shift_chunk[3] = 2
    shift_chunk[4] = 3
    shift_chunk[5] = 4
    shift_chunk[6] = 5
    shift_chunk[7] = 6
    shift_chunk[8:16] = shift_chunk[:8] + 8
    shift_chunk[16:24] = shift_chunk[:8] + 16
    shift_chunk[24:32] = shift_chunk[:8] + 24
    shift_chunk[32:40] = shift_chunk[:8] + 32
    shift_chunk[40:48] = shift_chunk[:8] + 40
    shift_chunk[48:56] = shift_chunk[:8] + 48
    shift_chunk[56:64] = shift_chunk[:8] + 56

    # Prepare Inputs
    # x = torch.ones(size=(total_len,), device="npu", dtype=dtype).contiguous()
    x = torch.linspace(1, total_len, steps=total_len, device="npu", dtype=dtype)
    # x[8:] = 10
    # x[16:] = 100
    # x[24:] = 1000
    # x[32:] = -1
    # x[40:] = -10
    # x[48:] = -100
    # x[56:] = -1000
    # x[64:] = 0

    s = torch.zeros_like(x)

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU JIT Kernel compilation
    file = "kernel_scan_v.cpp"
    scan_func = jit_compile(file)

    print(
        f"Testing NPU scan kernel: tile_size={tile_size}x{tile_size}, total_len={total_len} ({n_tiles} tiles)"
    )

    scan_func(x, s, shift_chunk, total_len, tile_size)

    torch.npu.synchronize()

    print("Comparing results...")
    print("NPU scan result:\n", s.cpu()[:128])
    print("Input:\n", x.cpu()[:128])
    print("Expected:\n", expected_scan[:128])

    assert torch.allclose(
        s.cpu(), expected_scan, rtol=1e-3, atol=1e-3
    ), "Scan results do not match expected values!"

    print("All results matched. Scan test passed successfully.\n")

    clean_up(file)


if __name__ == "__main__":
    test_scan()
