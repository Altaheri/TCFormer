import torch, time


# helper for universal latency‑measurement 
def measure_latency(model, input_shape, device="cuda:0", warmup=100, runs=500):
    """Return mean forward‑pass latency (ms) for a single sample.

    Parameters
    ----------
    model : torch.nn.Module  –  already on *any* device (will be moved inside)
    input_shape : tuple      –  e.g. (1, C, T)
    device : str             –  "cuda:0", "cpu", …
    warmup : int             –  number of warm‑up passes (not timed)
    runs : int               –  number of timed passes
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    dummy = torch.randn(*input_shape, device=dev)

    with torch.no_grad():
        # warm‑up – fill kernels / autotuner caches
        for _ in range(warmup):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # timed passes
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / runs  # seconds per pass

    return elapsed * 1e3  # → milliseconds
