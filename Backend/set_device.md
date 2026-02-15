# Device Configuration

The training can run on either **CPU** or **CUDA (GPU)**. Configure it via:

## Option 1: Edit config.py directly

In [config.py](config.py), change:
```python
DEVICE: str = os.getenv("DEVICE", "auto")  # "auto", "cuda", or "cpu"
```

- `"auto"` — Auto-detect (uses CUDA if available, otherwise CPU)
- `"cuda"` — Force GPU (will fail if no CUDA)
- `"cpu"` — Force CPU

## Option 2: Set environment variable

```powershell
# Use GPU
$env:DEVICE="cuda"; python -m training.train

# Use CPU
$env:DEVICE="cpu"; python -m training.train

# Auto-detect (default)
$env:DEVICE="auto"; python -m training.train
```

## Check your current setup

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Install CUDA-enabled PyTorch (if needed)

If you have an NVIDIA GPU but `torch.cuda.is_available()` returns `False`, install CUDA PyTorch:

```powershell
# Uninstall CPU version
pip uninstall -y torch torchvision torchaudio torch-geometric

# Install CUDA 12.x version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Reinstall torch-geometric for CUDA
pip install torch-geometric
```

**Expected speedup**: 10-50x faster training on GPU vs CPU for graph neural networks.
