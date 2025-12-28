# 🎵 M1 Singing Voice Conversion - Usage Guide
### Complete Guide to Training and Inference

This comprehensive guide covers all aspects of using the M1 Singing Voice Conversion system, from data preparation to final inference.

---

## 📋 **Quick Reference**

| Operation | Command | Time Estimate |
|-----------|---------|---------------|
| **Environment Test** | `python test_m1_pipeline.py --test-all` | 2-3 minutes |
| **Train Model** | `python m1_singing_trainer.py --input-dir vocals --singer-name "Artist"` | 2-8 hours |
| **Voice Conversion** | `python m1_singing_inference.py --source input.wav --model model.pt --output result.wav` | Real-time |
| **Batch Processing** | `python svc_preprocessing_pipeline.py --batch-process vocal_files/` | 1-2x real-time |

---

## 🎤 **Part 1: Data Preparation**

### **1.1 Audio Requirements**

**Supported Formats:**
- WAV (recommended): 16-bit, 24-bit, 32-bit float
- MP3: Any bitrate (will be converted)
- FLAC: Lossless compression supported
- M4A/AAC: Converted during processing

**Quality Recommendations:**
- **Sample Rate**: 44.1kHz or higher (will be processed at 48kHz)
- **Bit Depth**: 16-bit minimum, 24-bit preferred
- **Length**: 30 seconds to 10 minutes per file
- **Content**: Clean vocals with minimal background noise

### **1.2 Preparing Raw Audio Data**

#### **Organize Your Data**
```bash
# Create directory structure for your singer
mkdir -p raw_vocals/my_singer
cd raw_vocals/my_singer

# Add your audio files
# my_singer/
# ├── song1.wav
# ├── song2.wav
# ├── song3.mp3
# └── song4.flac
```

#### **Convert Mixed Audio Files**
```bash
# Convert all files to WAV format
for file in *.{mp3,flac,m4a}; do
    if [ -f "$file" ]; then
        ffmpeg -i "$file" -ar 44100 -ac 1 "${file%.*}.wav"
        echo "Converted: $file → ${file%.*}.wav"
    fi
done
```

#### **Quality Check**
```bash
# Check audio file properties
for file in *.wav; do
    echo "Checking: $file"
    ffprobe -v quiet -select_streams a:0 -show_entries stream=sample_rate,channels,duration -of csv=p=0 "$file"
done
```

### **1.3 Data Quality Guidelines**

**✅ Good Quality Audio:**
- Clear, isolated vocals
- Minimal background noise
- Consistent volume levels
- No clipping or distortion
- Various vocal expressions (soft, loud, emotional)

**❌ Avoid:**
- Heavy instrumental backing
- Multiple singers simultaneously
- Excessive reverb or effects
- Very short clips (<10 seconds)
- Heavily compressed audio

---

## 🏃‍♂️ **Part 2: Quick Start Workflow**

### **2.1 Basic Training Example**

```bash
# 1. Activate environment
conda activate rvc-singing

# 2. Navigate to project directory
cd ~/projects/MIMIC

# 3. Verify installation
python test_m1_pipeline.py --test-environment

# 4. Train with default settings
python m1_singing_trainer.py \
    --input-dir raw_vocals/my_singer \
    --singer-name "MySinger" \
    --memory 16
```

### **2.2 Basic Inference Example**

```bash
# Convert voice using trained model
python m1_singing_inference.py \
    --source input_vocals.wav \
    --model ./singing_models/models/MySinger/MySinger_final.pt \
    --output converted_vocals.wav
```

### **2.3 Expected Processing Times**

| Hardware | Training (1 hour audio) | Inference (1 song) |
|----------|---------------------------|---------------------|
| M1 Base (8GB) | 4-6 hours | 30-60 seconds |
| M1 Pro (16GB) | 2-4 hours | 15-30 seconds |
| M1 Max (32GB) | 1.5-3 hours | 10-20 seconds |

---

## 🎯 **Part 3: Advanced Training**

### **3.1 Custom Configuration**

#### **Memory-Optimized Training**
```bash
# For M1 Base with 8GB RAM
python m1_singing_trainer.py \
    --input-dir vocals \
    --singer-name "Singer" \
    --memory 8 \
    --force-preprocess
```

#### **High-Quality Training**
```python
# Create custom config file: custom_config.py
from singing_optimized_config import M1SingingConfig

class HighQualityConfig(M1SingingConfig):
    @property
    def audio_config(self):
        config = super().audio_config
        config.update({
            "hop_length": 32,      # Even finer resolution
            "win_length": 4096,    # Larger window
            "n_mels": 128,         # More mel bands
        })
        return config
    
    @property  
    def training_config(self):
        config = super().training_config
        config.update({
            "epochs": 500,         # More training
            "learning_rate": 5e-5, # Lower learning rate
            "save_every_epochs": 10, # More frequent saves
        })
        return config
```

### **3.2 Monitoring Training Progress**

#### **Real-time Log Monitoring**
```bash
# Monitor training logs in real-time
tail -f ./singing_models/logs/MySinger/training.log
```

#### **Performance Monitoring**
```bash
# Monitor system resources during training
htop

# Monitor GPU usage
sudo powermetrics --samplers gpu_power -n 1
```

#### **Training Metrics**
```bash
# Check training progress
python -c "
import json
from pathlib import Path

log_file = Path('./singing_models/logs/MySinger/training.log')
if log_file.exists():
    with open(log_file) as f:
        lines = f.readlines()[-20:]  # Last 20 lines
        for line in lines:
            if 'loss' in line.lower():
                print(line.strip())
"
```

### **3.3 Resume Training**

#### **From Checkpoint**
```bash
# Resume training from specific checkpoint
python m1_singing_trainer.py \
    --resume \
    --singer-name "MySinger" \
    --checkpoint checkpoint_epoch_100.pt
```

#### **From Latest Checkpoint**
```bash
# Resume from latest checkpoint automatically
python m1_singing_trainer.py \
    --resume \
    --singer-name "MySinger" \
    --checkpoint latest
```

### **3.4 Multi-Singer Training**

```bash
# Train multiple singers in sequence
singers=("Singer1" "Singer2" "Singer3")
for singer in "${singers[@]}"; do
    echo "Training: $singer"
    python m1_singing_trainer.py \
        --input-dir "raw_vocals/$singer" \
        --singer-name "$singer" \
        --memory 16
    
    # Clear cache between singers
    python -c "
    import torch, gc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    print('Memory cleared')
    "
done
```

---

## 🎨 **Part 4: Advanced Inference**

### **4.1 Pitch Shifting and Control**

#### **Basic Pitch Adjustment**
```bash
# Shift pitch up by 2 semitones
python m1_singing_inference.py \
    --source vocals.wav \
    --model singer.pt \
    --output higher_pitch.wav \
    --pitch-shift 2

# Shift pitch down by 3 semitones
python m1_singing_inference.py \
    --source vocals.wav \
    --model singer.pt \
    --output lower_pitch.wav \
    --pitch-shift -3
```

#### **Energy and Dynamics Control**
```bash
# Increase vocal energy/power
python m1_singing_inference.py \
    --source vocals.wav \
    --model singer.pt \
    --output energetic.wav \
    --energy-scale 1.3

# Softer, more intimate vocal
python m1_singing_inference.py \
    --source vocals.wav \
    --model singer.pt \
    --output soft.wav \
    --energy-scale 0.7
```

### **4.2 Batch Inference Processing**

#### **Process Multiple Files**
```bash
# Create batch processing script
cat << 'EOF' > batch_inference.sh
#!/bin/bash

MODEL="./singing_models/models/MySinger/MySinger_final.pt"
INPUT_DIR="input_vocals"
OUTPUT_DIR="converted_vocals"

mkdir -p "$OUTPUT_DIR"

for input_file in "$INPUT_DIR"/*.wav; do
    if [ -f "$input_file" ]; then
        filename=$(basename "$input_file" .wav)
        output_file="$OUTPUT_DIR/${filename}_converted.wav"
        
        echo "Converting: $input_file"
        python m1_singing_inference.py \
            --source "$input_file" \
            --model "$MODEL" \
            --output "$output_file"
        
        echo "✅ Completed: $output_file"
    fi
done
EOF

chmod +x batch_inference.sh
./batch_inference.sh
```

#### **Parallel Processing**
```bash
# Process multiple files in parallel (M1 Pro/Max)
find input_vocals -name "*.wav" | parallel -j 2 python m1_singing_inference.py \
    --source {} \
    --model singer.pt \
    --output converted_vocals/{/.}_converted.wav
```

### **4.3 Quality Enhancement Options**

#### **Force Preprocessing**
```bash
# Force reprocessing of source audio for best quality
python m1_singing_inference.py \
    --source input.wav \
    --model singer.pt \
    --output output.wav \
    --force-preprocess
```

#### **Custom Processing Pipeline**
```python
# Advanced inference with custom settings
from m1_singing_inference import M1SingingInference

# Initialize with custom config
inference = M1SingingInference("path/to/model.pt")

# Process with custom parameters
result = inference.infer(
    source_path="input.wav",
    output_path="output.wav",
    pitch_shift=1.5,
    energy_scale=1.2,
    force_preprocess=True
)

print(f"Conversion completed: {result}")
```

---

## 🔧 **Part 5: Configuration and Optimization**

### **5.1 Performance Tuning**

#### **Memory Optimization**
```python
# Edit singing_optimized_config.py for your hardware
def _calculate_optimal_batch_size(self) -> int:
    # Custom batch sizes based on your experience
    if self.max_memory_gb >= 32:
        return 6  # Conservative for stability
    elif self.max_memory_gb >= 16:
        return 4  # Balanced performance
    else:
        return 2  # Maximum stability
```

#### **Training Speed Optimization**
```python
# Aggressive memory management for faster training
training_config.update({
    "empty_cache_every": 20,        # More frequent clearing
    "num_workers": 1,               # Reduce overhead
    "persistent_workers": False,    # Save memory
})
```

### **5.2 Audio Quality Tuning**

#### **Preprocessing Quality Settings**
```python
# High-quality preprocessing settings
preprocessing_config = {
    "segment_size": 512,     # Larger segments for better quality
    "overlap": 0.25,         # More overlap for smooth transitions
    "denoise_strength": 0.8, # Stronger noise reduction
}
```

#### **RMVPE Tuning**
```python
# Fine-tune pitch detection for your audio
audio_config.update({
    "f0_min": 80,           # Adjust for voice range
    "f0_max": 800,          # Adjust for voice range  
    "hop_length": 32,       # Finer pitch resolution
})
```

---

## 📊 **Part 6: Quality Assessment**

### **6.1 Automated Quality Testing**

#### **Preprocessing Quality Check**
```bash
# Test preprocessing pipeline quality
python test_m1_pipeline.py --test-quality --audio-file your_test_file.wav
```

#### **Model Performance Evaluation**
```bash
# Comprehensive model testing
python test_m1_pipeline.py --test-all

# Specific inference testing
python -c "
from m1_singing_inference import M1SingingInference
import time

# Test inference speed
inference = M1SingingInference('path/to/model.pt')
start_time = time.time()

result = inference.infer('test_audio.wav', 'output.wav')
inference_time = time.time() - start_time

print(f'Inference time: {inference_time:.2f}s')
print(f'Real-time factor: {10.0/inference_time:.2f}x')  # Assuming 10s test audio
"
```

### **6.2 Manual Quality Assessment**

#### **Listening Test Checklist**
- [ ] **Pitch Accuracy**: Correct pitch conversion without artifacts
- [ ] **Timbre Quality**: Natural vocal characteristics preserved
- [ ] **Clarity**: Clear vocals without distortion
- [ ] **Naturalness**: Human-like vocal quality
- [ ] **Consistency**: Stable quality throughout the song

#### **Technical Measurements**
```python
# Measure audio quality metrics
import torchaudio
import torch

def analyze_audio_quality(original_path, converted_path):
    orig, sr = torchaudio.load(original_path)
    conv, _ = torchaudio.load(converted_path)
    
    # Ensure same length
    min_len = min(orig.shape[-1], conv.shape[-1])
    orig, conv = orig[..., :min_len], conv[..., :min_len]
    
    # Calculate SNR
    signal_power = torch.mean(conv**2)
    noise_power = torch.mean((orig - conv)**2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    print(f"SNR: {snr:.2f} dB")
    return snr

# Usage
snr = analyze_audio_quality("original.wav", "converted.wav")
```

---

## 🛠️ **Part 7: Troubleshooting Common Issues**

### **7.1 Training Issues**

#### **Training Not Starting**
```bash
# Check environment
python test_m1_pipeline.py --test-environment

# Check input data
ls -la raw_vocals/MySinger/
python -c "
import torchaudio
for file in ['song1.wav', 'song2.wav']:  # Your files
    try:
        waveform, sr = torchaudio.load(f'raw_vocals/MySinger/{file}')
        print(f'{file}: {sr}Hz, {waveform.shape[-1]/sr:.1f}s')
    except Exception as e:
        print(f'{file}: ERROR - {e}')
"
```

#### **Training Very Slow**
```bash
# Monitor system resources
top -pid $(pgrep -f "python.*trainer")

# Check MPS usage
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
"
```

### **7.2 Inference Issues**

#### **Poor Quality Output**
```bash
# Force preprocessing to ensure clean input
python m1_singing_inference.py \
    --source input.wav \
    --model model.pt \
    --output output.wav \
    --force-preprocess

# Check model and source compatibility
python -c "
import torch
model_data = torch.load('model.pt', map_location='cpu')
print('Model sample rate:', model_data.get('sample_rate', 'Unknown'))
print('Model config:', model_data.get('config', 'Missing'))
"
```

#### **Inference Errors**
```bash
# Debug inference step by step
python -c "
from m1_singing_inference import M1SingingInference
import logging

logging.basicConfig(level=logging.DEBUG)
inference = M1SingingInference('model.pt')

# Test each step
dry_audio = inference.preprocess_source_audio('input.wav')
features = inference.extract_features(dry_audio)
converted = inference.convert_voice(features)
final = inference.post_process_audio(converted)
print('All steps completed successfully')
"
```

---

## 📚 **Part 8: Best Practices**

### **8.1 Data Management**

#### **Organized Directory Structure**
```
MIMIC/
├── raw_vocals/
│   ├── Singer1/
│   ├── Singer2/
│   └── Singer3/
├── singing_models/
│   ├── models/
│   ├── checkpoints/
│   └── logs/
├── processed_audio/
├── converted_outputs/
└── test_outputs/
```

#### **Version Control for Models**
```bash
# Tag model versions
cd singing_models/models/MySinger
git init
git add MySinger_final.pt
git commit -m "MySinger v1.0 - Initial training"
git tag v1.0

# Track model performance
echo "SNR: 12.5dB, Training time: 3h" > MySinger_v1.0_metrics.txt
```

### **8.2 Workflow Optimization**

#### **Efficient Training Pipeline**
```bash
# Complete training workflow
#!/bin/bash
SINGER_NAME="$1"
INPUT_DIR="raw_vocals/$SINGER_NAME"

echo "🎵 Training $SINGER_NAME"

# 1. Verify data quality
python test_m1_pipeline.py --test-preprocessing --audio-file "$INPUT_DIR/sample.wav"

# 2. Train model
python m1_singing_trainer.py --input-dir "$INPUT_DIR" --singer-name "$SINGER_NAME"

# 3. Test inference
python m1_singing_inference.py \
    --source "$INPUT_DIR/test_sample.wav" \
    --model "singing_models/models/$SINGER_NAME/${SINGER_NAME}_final.pt" \
    --output "test_outputs/${SINGER_NAME}_test.wav"

echo "✅ Training complete for $SINGER_NAME"
```

---

## 🎯 **Part 9: Production Deployment**

### **9.1 Model Export and Optimization**

#### **Export Optimized Model**
```python
# Optimize model for inference
import torch
from pathlib import Path

def optimize_model_for_inference(model_path, output_path):
    """Optimize trained model for faster inference."""
    model_data = torch.load(model_path, map_location='cpu')
    
    # Remove training-only data
    optimized_data = {
        'singer_name': model_data['singer_name'],
        'config': model_data['config'],
        'sample_rate': model_data['sample_rate'],
        'hop_length': model_data['hop_length'], 
        'f0_method': model_data['f0_method'],
        # Add only inference-needed weights
    }
    
    torch.save(optimized_data, output_path)
    
    # Compare sizes
    original_size = Path(model_path).stat().st_size / 1024 / 1024
    optimized_size = Path(output_path).stat().st_size / 1024 / 1024
    
    print(f"Original: {original_size:.1f}MB")
    print(f"Optimized: {optimized_size:.1f}MB") 
    print(f"Reduction: {(1-optimized_size/original_size)*100:.1f}%")

# Usage
optimize_model_for_inference(
    "singing_models/models/MySinger/MySinger_final.pt",
    "production_models/MySinger_inference.pt"
)
```

### **9.2 API Integration**

#### **Simple REST API**
```python
# api_server.py
from flask import Flask, request, send_file
from m1_singing_inference import M1SingingInference
import tempfile
import os

app = Flask(__name__)
inference_engine = M1SingingInference("production_models/singer.pt")

@app.route('/convert', methods=['POST'])
def convert_voice():
    if 'audio' not in request.files:
        return {"error": "No audio file provided"}, 400
    
    audio_file = request.files['audio']
    pitch_shift = float(request.form.get('pitch_shift', 0))
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
        audio_file.save(tmp_input.name)
        
        # Convert voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
            result = inference_engine.infer(
                tmp_input.name,
                tmp_output.name,
                pitch_shift=pitch_shift
            )
            
            # Return converted audio
            return send_file(tmp_output.name, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📞 **Support and Resources**

### **Getting Help**
1. **Documentation**: Check `README.md` and `M1_TROUBLESHOOTING_GUIDE.md`
2. **Diagnostics**: Run `python test_m1_pipeline.py --test-all`
3. **Logs**: Check `singing_models/logs/` for detailed information
4. **Community**: Share experiences and get help from other users

### **Performance Tips**
- **Memory**: Use appropriate batch sizes for your hardware
- **Quality**: Higher hop_length = better quality but slower processing
- **Training**: More epochs generally improve quality but take longer
- **Data**: Higher quality input data = better output results

---

**🎵 Ready to create amazing singing voice conversions! 🎵**

*For technical details, see the complete documentation suite including README.md, INSTALLATION_GUIDE.md, and M1_TROUBLESHOOTING_GUIDE.md*