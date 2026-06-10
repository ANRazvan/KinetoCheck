# PyTorch CUDA Setup Status - KinetoCheck

**Last Updated**: March 26, 2026
**Verified**: ✓ GPU Hardware Ready, ✗ CUDA/PyTorch Blocked by Network

## Hardware Status

### GPU Information
```
GPU: NVIDIA GeForce GTX 1650
Memory: 4096 MB (4 GB)
Driver Version: 566.07
CUDA Toolkit: 12.7
Status: READY
```

**verified via**: `nvidia-smi`

### Python Environment
```
Python Version: 3.13.1
Environment Type: Virtual Environment (.venv)
Location: d:\Programming\KinetoCheck\.venv
```

## Installed Packages

| Package | Version | CUDA Support | Status |
|---------|---------|--------------|--------|
| torch | 2.11.0 | ✗ CPU-only | Installed |
| torchvision | 0.26.0 | ✗ CPU-only | Installed |
| torchaudio | 2.11.0 | ✗ CPU-only | Installed |
| torch-geometric | 2.6.1 | ✓ (works with CPU) | Installed |

## Issue & Resolution

### Problem
- PyTorch installed as CPU-only version (`2.11.0+cpu`)
- Network cannot access PyTorch CUDA wheel servers
- GPU available but not being used

### Root Cause
- `https://download.pytorch.org/whl/cu121` returns "No matching distribution found"
- This suggests corporate proxy/firewall restrictions or network connectivity issue

### Solutions

#### Solution A: Train on CPU (Works Now)
**Pros**: Works immediately
**Cons**: ~10-50x slower training, not utilizing GPU

```bash
cd Backend
python -m temporal_pyramid_stgat.training.train_triplet --exercise 0 --batch-size 8 --epochs 10
```

**For faster results on CPU**, reduce batch size:
```bash
python -m temporal_pyramid_stgat.training.train_triplet --exercise 0 --batch-size 4 --epochs 50
```

#### Solution B: Fix Network Access (Requires IT/Admin)
1. Request IT to whitelist: `download.pytorch.org`
2. Or disable proxy for PyPI: Contact IT
3. Then run:
```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu127
```

#### Solution C: Manual Wheel Installation
1. Download wheels manually from PyTorch website on a different network
2. Transfer to your machine
3. Install locally: `pip install torch-2.1.0+cu121.whl`

#### Solution D: Use Miniconda (If Standard Conda Not Available)
1. Install Miniconda from `https://docs.conda.io/projects/miniconda/en/latest/`
2. Create environment: `conda create -n kineto python=3.11`
3. Install PyTorch: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

## Workaround: Estimate Training Performance

### CPU Performance (Current)
- **GPU Training**: ~100-500 samples/sec
- **CPU Training** (Core i7+): ~1-5 samples/sec
- **For UI-PRMD (~900 sequences)**: 
  - GPU: ~2-10 minutes per epoch
  - CPU: 3-15 minutes per epoch

### Recommended CPU Settings
```bash
# Reduce batch size to fit in RAM
python -m temporal_pyramid_stgat.training.train_triplet \
  --exercise 0 \
  --batch-size 4 \  # Reduced from 32
  --epochs 50 \      # Reduced from 100
  --lr 0.001
```

## Next Steps

1. **Option 1**: Start training on CPU now (accept slower speed)
   ```bash
   python -m temporal_pyramid_stgat.training.train_triplet --exercise 0 --batch-size 4 --epochs 50
   ```

2. **Option 2**: Contact IT to fix PyTorch server access, then re-install CUDA PyTorch

3. **Option 3**: Install Miniconda and use conda for package management

## Verification Commands

Check current PyTorch setup:
```bash
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
```

Check GPU availability:
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Check NVIDIA GPU directly:
```bash
nvidia-smi
```

## Contact for Support

If you need to fix this:
1. Contact your network IT department about PyTorch server access
2. Or use Miniconda (conda-based installation often works better in restricted networks)
3. Or train on CPU (accept slower speed)

---

**Status**: Ready to train on CPU. Awaiting network resolution for GPU CUDA support.
