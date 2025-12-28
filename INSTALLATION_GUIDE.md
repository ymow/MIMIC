# 🔧 M1 Singing Voice Conversion - Complete Installation Guide
### Step-by-Step Setup for Apple Silicon (M1/M2/M3)

This guide provides comprehensive installation instructions for setting up the M1 Singing Voice Conversion system on Apple Silicon hardware.

---

## 📋 **Prerequisites Check**

Before starting the installation, verify your system meets these requirements:

```bash
# Check system architecture (should be arm64)
uname -m

# Check macOS version (should be 12.3+)
sw_vers

# Check available memory (16GB+ recommended)
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'

# Check available storage (50GB+ recommended)
df -h
```

**Minimum Requirements:**
- ✅ Apple Silicon Mac (M1/M2/M3)
- ✅ macOS 12.3+ (for MPS support)
- ✅ 8GB+ RAM (16GB+ recommended)
- ✅ 50GB+ free storage
- ✅ Stable internet connection

---

## 🛠️ **Step 1: System Dependencies**

### **1.1 Install Xcode Command Line Tools**
```bash
# Install Xcode command line tools
xcode-select --install

# Verify installation
xcode-select --version
```

### **1.2 Install Homebrew**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (for Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Verify installation
brew --version
```

### **1.3 Install FFmpeg and Audio Tools**
```bash
# Install FFmpeg with all codecs
brew install ffmpeg

# Install additional audio tools
brew install sox libsndfile

# Verify FFmpeg installation
ffmpeg -version | head -1
```

---

## 🐍 **Step 2: Python Environment Setup**

### **2.1 Download Miniconda for Apple Silicon**
```bash
# Download Miniconda for Apple Silicon
cd ~/Downloads
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda
bash Miniconda3-latest-MacOSX-arm64.sh -b -p ~/miniconda3

# Initialize conda
~/miniconda3/bin/conda init zsh  # or 'bash' if using bash
source ~/.zshrc  # or ~/.bashrc for bash

# Verify conda installation
conda --version
```

### **2.2 Create RVC Singing Environment**
```bash
# Create dedicated environment for singing voice conversion
conda create -n rvc-singing python=3.10 -y

# Activate environment
conda activate rvc-singing

# Verify Python version
python --version  # Should show Python 3.10.x
```

---

## 🔥 **Step 3: PyTorch with MPS Support**

### **3.1 Install PyTorch for Apple Silicon**
```bash
# Activate environment
conda activate rvc-singing

# Install PyTorch with MPS support
conda install pytorch torchaudio -c pytorch -y

# Verify MPS availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Expected Output:**
```
PyTorch version: 2.1.0
MPS available: True
```

### **3.2 Test MPS Functionality**
```bash
# Test MPS with simple operation
python -c "
import torch
device = torch.device('mps')
x = torch.randn(100, 100, device=device)
y = torch.mm(x, x.T)
print(f'MPS test successful! Result shape: {y.shape}')
"
```

---

## 📦 **Step 4: Core Dependencies**

### **4.1 Install Audio Processing Libraries**
```bash
# Activate environment
conda activate rvc-singing

# Install core audio dependencies
pip install librosa==0.10.2
pip install soundfile
pip install resampy
pip install numba

# Install scientific computing
pip install numpy scipy
pip install scikit-learn
```

### **4.2 Install Faiss for Apple Silicon**
```bash
# CRITICAL: Install CPU-only version to avoid segfaults
conda install -c conda-forge faiss-cpu -y

# Verify Faiss installation
python -c "import faiss; print(f'Faiss version: {faiss.__version__}')"
```

### **4.3 Install Audio Separator**
```bash
# Install audio-separator for vocal isolation
pip install audio-separator

# Verify installation
audio-separator --help | head -5
```

---

## 📂 **Step 5: Project Setup**

### **5.1 Clone and Setup Project**
```bash
# Navigate to your projects directory
cd ~/projects  # or your preferred location

# Clone the repository (replace with actual URL)
git clone <repository-url> MIMIC
cd MIMIC

# Install project dependencies
pip install -r requirements.txt
```

### **5.2 Create Required Directories**
```bash
# Create directory structure
mkdir -p singing_models/{models,logs,checkpoints,datasets,outputs}
mkdir -p test_outputs
mkdir -p test_logs

# Verify directory structure
tree -L 2 singing_models/
```

---

## 🤖 **Step 6: Download Pre-trained Models**

### **6.1 Setup Model Cache Directory**
```bash
# Create cache directory for models
mkdir -p ~/.cache/audio-separator

# Navigate to cache directory
cd ~/.cache/audio-separator
```

### **6.2 Download BS-Roformer Model**
```bash
# Download BS-Roformer for instrument separation
curl -L -o "model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Verify download
ls -lh model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

### **6.3 Download Mel-Band Roformer Model**
```bash
# Download Mel-Band Roformer for dereverberation
curl -L -o "mel_band_roformer_karaoke.ckpt" \
"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mel_band_roformer_karaoke.ckpt"

# Verify download
ls -lh mel_band_roformer_karaoke.ckpt
```

### **6.4 Test Model Downloads**
```bash
# Return to project directory
cd ~/projects/MIMIC  # or your project path

# Test audio separator with downloaded models
echo "Testing model downloads..."
python -c "
from svc_preprocessing_pipeline import SVCPreprocessingPipeline
pipeline = SVCPreprocessingPipeline()
print('✅ Models loaded successfully!')
"
```

---

## 🧪 **Step 7: Installation Verification**

### **7.1 Run Comprehensive Environment Test**
```bash
# Navigate to project directory
cd ~/projects/MIMIC

# Run complete environment verification
python test_m1_pipeline.py --test-environment
```

**Expected Output:**
```
🔍 Testing M1 environment compatibility...
  ✅ Apple Silicon detected
  ✅ MPS backend available
  ✅ MPS computation test passed
  ✅ System memory: 16.0GB
  ✅ Core dependencies available
  ✅ Audio backend configured
🎉 Environment compatibility: PASSED
```

### **7.2 Test Preprocessing Pipeline**
```bash
# Generate test audio and verify preprocessing
python test_m1_pipeline.py --test-preprocessing

# Expected: Processing should complete without errors
```

### **7.3 Test Memory Management**
```bash
# Test M1-specific memory management
python test_m1_pipeline.py --test-memory

# Expected: Memory usage should remain stable
```

---

## 🎵 **Step 8: Quick Functionality Test**

### **8.1 Create Test Audio**
```bash
# Create a simple test audio file
python -c "
import torch
import torchaudio
import numpy as np

# Generate test sine wave (A4 = 440Hz)
duration = 5  # seconds
sample_rate = 48000
t = torch.linspace(0, duration, sample_rate * duration)
audio = 0.5 * torch.sin(2 * np.pi * 440 * t)

# Save test audio
torchaudio.save('test_audio.wav', audio.unsqueeze(0), sample_rate)
print('✅ Test audio created: test_audio.wav')
"
```

### **8.2 Test Preprocessing Pipeline**
```bash
# Test the cascaded preprocessing pipeline
python svc_preprocessing_pipeline.py --test-mode --input test_audio.wav --output test_processed

# Expected: Should create processed audio file
ls -la test_processed/
```

### **8.3 Test Configuration System**
```bash
# Test M1-optimized configuration
python -c "
from singing_optimized_config import create_config_for_model
config = create_config_for_model('test_model', memory_gb=16.0)
print('✅ Configuration system working!')
print(f'   Batch size: {config.batch_size}')
print(f'   Sample rate: {config.audio_config[\"sample_rate\"]}Hz')
"
```

---

## 🎯 **Step 9: Final Validation**

### **9.1 Run Complete Test Suite**
```bash
# Run comprehensive validation
python test_m1_pipeline.py --test-all

# This will test:
# - Environment compatibility
# - Preprocessing pipeline
# - Memory management 
# - Audio quality metrics
# - Performance benchmarks
```

### **9.2 Check Installation Summary**
```bash
# Generate installation report
python -c "
import sys
import torch
import torchaudio
import platform
import psutil

print('🎵 M1 Singing Voice Conversion - Installation Summary')
print('=' * 60)
print(f'System: {platform.system()} {platform.release()}')
print(f'Architecture: {platform.machine()}')
print(f'Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB')
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'TorchAudio: {torchaudio.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
print('=' * 60)
print('✅ Installation completed successfully!')
print('🎵 Ready for singing voice conversion!')
"
```

---

## 🚨 **Troubleshooting Installation Issues**

### **Common Installation Problems**

#### **Issue 1: Conda Command Not Found**
```bash
# Add conda to PATH
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### **Issue 2: MPS Not Available**
```bash
# Check macOS version (needs 12.3+)
sw_vers

# Update macOS if needed
softwareupdate -l

# Reinstall PyTorch
pip uninstall torch torchaudio -y
conda install pytorch torchaudio -c pytorch -y
```

#### **Issue 3: Faiss Segmentation Fault**
```bash
# Remove GPU version and install CPU-only
pip uninstall faiss-gpu faiss-cpu -y
conda install -c conda-forge faiss-cpu -y
```

#### **Issue 4: Audio Separator Installation Failed**
```bash
# Install FFmpeg first
brew install ffmpeg

# Reinstall with force
pip install audio-separator --force-reinstall --no-cache-dir
```

#### **Issue 5: Model Download Failed**
```bash
# Manual download with curl
mkdir -p ~/.cache/audio-separator
cd ~/.cache/audio-separator

# Download with retry
curl --retry 3 -L -o "model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
```

---

## 📋 **Installation Checklist**

Mark each item as completed:

- [ ] ✅ **System Requirements**: Apple Silicon, macOS 12.3+, 16GB+ RAM
- [ ] ✅ **Xcode Tools**: `xcode-select --install` completed
- [ ] ✅ **Homebrew**: Installed and PATH configured  
- [ ] ✅ **FFmpeg**: `brew install ffmpeg` completed
- [ ] ✅ **Miniconda**: Apple Silicon version installed
- [ ] ✅ **Python Environment**: `rvc-singing` environment created
- [ ] ✅ **PyTorch MPS**: Installed and MPS available
- [ ] ✅ **Faiss CPU**: Installed without segfaults
- [ ] ✅ **Audio Separator**: CLI tool working
- [ ] ✅ **Project Setup**: Repository cloned and dependencies installed
- [ ] ✅ **Pre-trained Models**: BS-Roformer and Mel-Band models downloaded
- [ ] ✅ **Environment Test**: `test_m1_pipeline.py --test-environment` passed
- [ ] ✅ **Pipeline Test**: Preprocessing pipeline working
- [ ] ✅ **Full Validation**: Complete test suite passed

---

## 🎉 **Next Steps**

Once installation is complete:

1. **Train Your First Model**: See training examples in README.md
2. **Process Audio**: Use the preprocessing pipeline for vocal isolation
3. **Run Inference**: Convert voices using trained models
4. **Optimize Performance**: Adjust configurations for your hardware

---

## 📞 **Support**

If you encounter issues during installation:

1. **Check Prerequisites**: Verify system requirements
2. **Run Diagnostics**: Use `test_m1_pipeline.py --test-environment` 
3. **Review Logs**: Check error messages for specific issues
4. **Consult Troubleshooting**: See `M1_TROUBLESHOOTING_GUIDE.md`

---

**🎵 Installation Complete - Ready to Create Amazing Singing Voice Conversions! 🎵**