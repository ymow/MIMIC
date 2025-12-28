# M1 Singing Voice Conversion - Troubleshooting Guide
## Apple Silicon Specific Issues and Solutions

### 🚨 Common M1/M2/M3 Issues and Solutions

This guide addresses the most frequent issues encountered when running our M1-optimized singing voice conversion pipeline on Apple Silicon hardware.

---

## 📋 Quick Diagnostics Checklist

Before troubleshooting specific issues, run our automated diagnostic:

```bash
python test_m1_pipeline.py --test-environment
```

This will verify:
- ✅ Apple Silicon detection
- ✅ MPS backend availability  
- ✅ Memory configuration
- ✅ Required dependencies
- ✅ Audio backend setup

---

## 🔧 Environment Setup Issues

### Issue 1: Conda Environment Broken/Corrupted

**Symptoms:**
- `conda: command not found`
- `conda activate` fails
- Python packages not found

**Solution:**
```bash
# Remove existing conda installation
rm -rf ~/miniconda3

# Download fresh Miniconda for Apple Silicon
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p ~/miniconda3

# Initialize conda
~/miniconda3/bin/conda init zsh  # or bash
source ~/.zshrc

# Create fresh environment
conda create -n rvc-singing python=3.10 -y
conda activate rvc-singing
```

### Issue 2: PyTorch MPS Not Available

**Symptoms:**
- `torch.backends.mps.is_available()` returns `False`
- Training runs on CPU only
- Very slow performance

**Solution:**
```bash
# Install PyTorch with MPS support
conda install pytorch torchaudio -c pytorch -y

# Verify installation
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**If still failing:**
```bash
# Update macOS (MPS requires macOS 12.3+)
softwareupdate -l

# Reinstall PyTorch nightly
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Issue 3: Faiss Segmentation Fault

**Symptoms:**
- `Segmentation fault: 11` when importing faiss
- Crashes during feature extraction
- OpenMP related errors

**Solution:**
```bash
# Remove any existing faiss installation
pip uninstall faiss-gpu faiss-cpu -y

# Install CPU-only version for M1
conda install -c conda-forge faiss-cpu -y

# Alternative: Use OpenBLAS version
pip install faiss-cpu --no-binary faiss-cpu
```

**Prevention:**
Always use `faiss-cpu` on Apple Silicon, never `faiss-gpu`.

---

## 🎵 Audio Processing Issues

### Issue 4: Audio-Separator Installation Problems

**Symptoms:**
- `audio-separator` command not found
- FFmpeg errors during installation
- Compilation failures

**Solution:**
```bash
# Install system dependencies first
brew install ffmpeg

# Install audio-separator with CPU optimization
pip install audio-separator --force-reinstall

# Verify installation
audio-separator --help
```

### Issue 5: Model Download Failures

**Symptoms:**
- "Model not found" errors
- Incomplete downloads
- Corrupted model files

**Solution:**
```bash
# Clear model cache
rm -rf ~/.cache/audio-separator

# Manual model download
mkdir -p ~/.cache/audio-separator
cd ~/.cache/audio-separator

# Download BS-Roformer
curl -L -o "model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Download Mel-Band Roformer  
curl -L -o "mel_band_roformer_karaoke.ckpt" \
"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mel_band_roformer_karaoke.ckpt"
```

### Issue 6: Preprocessing Pipeline Failures

**Symptoms:**
- "Dry vocal extraction failed" 
- Empty output files
- Audio corruption

**Solution:**
```bash
# Test pipeline with verbose logging
python svc_preprocessing_pipeline.py --test-mode --verbose

# Check audio file format
ffprobe your_audio.wav

# Convert to supported format if needed
ffmpeg -i input.mp3 -ar 44100 -ac 2 output.wav
```

---

## 💾 Memory and Performance Issues

### Issue 7: Out of Memory Errors

**Symptoms:**
- "RuntimeError: MPS backend out of memory"
- System freezing during training
- Swap file creation

**Solution:**

**Immediate fix:**
```python
# In your training script
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

import gc
gc.collect()
```

**Configuration fix:**
```python
# Reduce batch size in singing_optimized_config.py
def _calculate_optimal_batch_size(self) -> int:
    if self.max_memory_gb >= 32:
        return 4  # Reduced from 8
    elif self.max_memory_gb >= 16:
        return 3  # Reduced from 6  
    else:
        return 2  # Reduced from 4
```

### Issue 8: Very Slow Training Speed

**Symptoms:**
- Training much slower than expected
- High CPU usage, low GPU usage
- Frequent cache clearing

**Solution:**

**Check MPS usage:**
```bash
# Monitor GPU activity
sudo powermetrics --samplers gpu_power -n 1
```

**Optimize configuration:**
```python
# In training config
"num_workers": 1,  # Reduce from 2
"pin_memory": False,  # Keep disabled for M1
"persistent_workers": False,  # Add this
```

### Issue 9: Memory Leaks During Training

**Symptoms:**
- Memory usage steadily increasing
- Eventually runs out of memory
- System becomes unresponsive

**Solution:**

**Implement aggressive memory management:**
```python
# Add to training loop
if step % 10 == 0:  # More frequent clearing
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    # Additional M1 specific cleanup
    import os
    os.system('purge')  # macOS memory purge
```

---

## 🎤 Audio Quality Issues

### Issue 10: Poor Vocal Isolation Quality

**Symptoms:**
- Instruments bleeding through
- Distorted vocals
- Artifacts in dry signal

**Solution:**

**Adjust preprocessing parameters:**
```python
# In svc_preprocessing_pipeline.py
"segment_size": 512,  # Increase for better quality
"overlap": 0.25,      # Increase overlap
"denoise_strength": 0.7,  # Adjust denoising
```

**Use higher quality models:**
```bash
# Download higher quality models
audio-separator --model_name="UVR-MDX-NET-Inst_HQ_3" your_audio.wav
```

### Issue 11: Pitch Detection Issues with RMVPE

**Symptoms:**
- Incorrect pitch extraction
- Unstable F0 contours  
- Poor singing quality

**Solution:**

**RMVPE configuration:**
```python
# In audio config
"f0_method": "rmvpe",
"f0_min": 80,      # Increase minimum for singing
"f0_max": 800,     # Adjust maximum
"hop_length": 32,  # Even finer resolution
```

**Fallback F0 method:**
```python
# If RMVPE fails, use harvest
"f0_method": "harvest",
"f0_min": 50,
"f0_max": 1100,
```

---

## 🔍 Debugging and Diagnostics

### Issue 12: Silent Failures in Pipeline

**Symptoms:**
- Scripts complete without errors
- But output files are empty or corrupted
- No clear error messages

**Solution:**

**Enable comprehensive logging:**
```bash
# Run with maximum verbosity
export PYTHONPATH=$PWD
python -u m1_singing_trainer.py --verbose --debug 2>&1 | tee training.log
```

**Add debugging code:**
```python
# In your scripts
import logging
logging.basicConfig(level=logging.DEBUG)

# Add file size checks
def verify_audio_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file missing: {path}")
    
    size_mb = os.path.getsize(path) / 1024 / 1024
    if size_mb < 0.1:
        raise ValueError(f"Audio file too small: {size_mb:.2f}MB")
    
    print(f"✅ Audio file OK: {path} ({size_mb:.2f}MB)")
```

### Issue 13: Model Training Not Converging

**Symptoms:**
- Loss not decreasing
- Training stuck at high loss values
- No improvement over epochs

**Solution:**

**Check data quality:**
```bash
# Validate processed dataset
python test_m1_pipeline.py --test-quality --audio-file processed_vocals/
```

**Adjust learning parameters:**
```python
# In training config
"learning_rate": 2e-5,  # Reduce learning rate
"warmup_steps": 1000,   # Add warmup
"gradient_clip_val": 1.0,  # Add gradient clipping
```

---

## 🚀 Performance Optimization Tips

### Best Practices for M1 Training

1. **Memory Management:**
   ```python
   # Clear cache every N steps
   if step % cache_clear_interval == 0:
       torch.mps.empty_cache()
       gc.collect()
   ```

2. **Batch Size Optimization:**
   ```python
   # Start small and increase gradually
   batch_sizes = [2, 4, 6, 8]
   for bs in batch_sizes:
       try:
           model.train_batch(batch_size=bs)
           break
       except RuntimeError:
           continue
   ```

3. **Monitoring:**
   ```bash
   # Monitor system resources
   htop
   watch -n 1 'ps aux | grep python'
   ```

### Hardware-Specific Configurations

**M1 Base (8GB):**
```python
batch_size = 2
num_workers = 0
cache_clear_every = 5
```

**M1 Pro (16GB):**
```python
batch_size = 4
num_workers = 1  
cache_clear_every = 10
```

**M1 Max/Ultra (32GB+):**
```python
batch_size = 6
num_workers = 2
cache_clear_every = 20
```

---

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Run Diagnostics:**
   ```bash
   python test_m1_pipeline.py --test-all > diagnostic_report.txt
   ```

2. **Collect System Information:**
   ```bash
   system_profiler SPHardwareDataType
   uname -a
   python --version
   pip list > requirements_current.txt
   ```

3. **Check Logs:**
   ```bash
   # Training logs
   tail -f ./singing_models/logs/training.log
   
   # System logs for crashes
   log show --predicate 'process == "Python"' --last 1h
   ```

4. **Create Minimal Reproduction:**
   ```bash
   # Test with minimal example
   python test_m1_pipeline.py --test-preprocessing --audio-file test_audio.wav
   ```

---

## 📚 Additional Resources

- **Apple Silicon PyTorch Guide:** https://pytorch.org/docs/stable/notes/mps.html
- **M1 Memory Architecture:** Understanding unified memory constraints
- **Audio Processing on M1:** Best practices for audio workloads
- **Conda on Apple Silicon:** Official installation guide

---

**💡 Pro Tip:** Always test with smaller datasets first before running full training to identify issues early and save time.

**⚠️ Remember:** M1 chips have unified memory architecture - both CPU and GPU share the same memory pool, so memory management is critical for stable training.