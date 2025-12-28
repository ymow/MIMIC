#!/usr/bin/env python3
"""
Simple Enhanced M2M Voice Builder
=================================

Focused solution to create natural, non-mechanical M2M voice output.
Simplified approach with effective results.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
import subprocess
import tempfile
import time

class SimpleEnhancedVoiceBuilder:
    """Simple but effective M2M voice builder"""
    
    def __init__(self):
        self.setup_logging()
        self.sample_rate = 48000
        self.output_dir = Path("simple_enhanced_output")
        self.output_dir.mkdir(exist_ok=True)
        self.m2m_source = "test_results/separation/M2M_Pretty_Boy_(Vocals)_UVR_MDXNET_KARA.wav"
        self.logger.info("Simple Enhanced Voice Builder initialized")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def create_enhanced_speech(self, text: str, output_filename: str, emotion_style: str = "natural") -> bool:
        """Create enhanced M2M speech"""
        
        self.logger.info(f"🎭 Creating enhanced speech ({emotion_style}): {output_filename}")
        
        try:
            # Create enhanced SSML
            ssml_text = self._create_enhanced_ssml(text, emotion_style)
            
            # Generate base audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            # Use edge-tts with enhanced parameters
            result = subprocess.run([
                "edge-tts",
                "--voice", "en-US-AriaNeural",
                "--text", ssml_text,
                "--write-media", temp_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and Path(temp_path).exists():
                # Apply M2M conversion
                output_path = self.output_dir / output_filename
                success = self._convert_to_m2m(temp_path, str(output_path), "speech")
                
                # Cleanup
                Path(temp_path).unlink()
                
                if success:
                    self.logger.info(f"   ✅ Enhanced speech created: {output_filename}")
                    return True
                else:
                    self.logger.error(f"   ❌ M2M conversion failed")
                    return False
            else:
                self.logger.error(f"   ❌ TTS generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"   ❌ Error creating enhanced speech: {e}")
            return False
    
    def create_enhanced_singing(self, lyrics: str, output_filename: str, emotion_style: str = "melodic") -> bool:
        """Create enhanced M2M singing"""
        
        self.logger.info(f"🎵 Creating enhanced singing ({emotion_style}): {output_filename}")
        
        try:
            # Create singing SSML
            ssml_text = self._create_singing_ssml(lyrics, emotion_style)
            
            # Generate base audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            result = subprocess.run([
                "edge-tts",
                "--voice", "en-US-AriaNeural",
                "--text", ssml_text,
                "--write-media", temp_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and Path(temp_path).exists():
                # Apply M2M conversion
                output_path = self.output_dir / output_filename
                success = self._convert_to_m2m(temp_path, str(output_path), "singing")
                
                # Cleanup
                Path(temp_path).unlink()
                
                if success:
                    self.logger.info(f"   ✅ Enhanced singing created: {output_filename}")
                    return True
                else:
                    self.logger.error(f"   ❌ M2M conversion failed")
                    return False
            else:
                self.logger.error(f"   ❌ TTS generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"   ❌ Error creating enhanced singing: {e}")
            return False
    
    def _create_enhanced_ssml(self, text: str, emotion_style: str) -> str:
        """Create enhanced SSML for speech"""
        
        # Emotion configurations
        configs = {
            "natural": {"style": "friendly", "degree": "1.0", "rate": "95%", "pitch": "medium"},
            "emotional": {"style": "empathetic", "degree": "1.8", "rate": "90%", "pitch": "medium"},
            "excited": {"style": "excited", "degree": "2.0", "rate": "105%", "pitch": "medium"},
            "happy": {"style": "cheerful", "degree": "1.5", "rate": "100%", "pitch": "medium"},
            "tender": {"style": "gentle", "degree": "1.5", "rate": "85%", "pitch": "low"},
            "confident": {"style": "confident", "degree": "1.3", "rate": "100%", "pitch": "medium"}
        }
        
        config = configs.get(emotion_style, configs["natural"])
        
        # Add natural phrasing
        enhanced_text = self._add_speech_phrasing(text)
        
        ssml = f"""<speak version="1.0" xml:lang="en-US">
<mstts:express-as style="{config['style']}" styledegree="{config['degree']}">
<prosody rate="{config['rate']}" pitch="{config['pitch']}" volume="85%">
{enhanced_text}
</prosody>
</mstts:express-as>
</speak>"""
        
        return ssml
    
    def _create_singing_ssml(self, lyrics: str, emotion_style: str) -> str:
        """Create enhanced SSML for singing"""
        
        # Singing configurations
        configs = {
            "melodic": {"style": "lyrical", "degree": "1.3", "rate": "85%", "pitch": "medium"},
            "emotional": {"style": "empathetic", "degree": "2.0", "rate": "80%", "pitch": "medium"},
            "ballad": {"style": "sad", "degree": "1.5", "rate": "75%", "pitch": "low"},
            "upbeat": {"style": "cheerful", "degree": "1.5", "rate": "95%", "pitch": "medium"}
        }
        
        config = configs.get(emotion_style, configs["melodic"])
        
        # Add musical phrasing
        enhanced_lyrics = self._add_musical_phrasing(lyrics)
        
        ssml = f"""<speak version="1.0" xml:lang="en-US">
<mstts:express-as style="{config['style']}" styledegree="{config['degree']}">
<prosody rate="{config['rate']}" pitch="{config['pitch']}" volume="85%">
{enhanced_lyrics}
</prosody>
</mstts:express-as>
</speak>"""
        
        return ssml
    
    def _add_speech_phrasing(self, text: str) -> str:
        """Add natural speech phrasing"""
        import re
        
        # Add pauses
        text = re.sub(r'([.!?])\s+', r'\1<break time="800ms"/> ', text)
        text = re.sub(r'([,;:])\s+', r'\1<break time="300ms"/> ', text)
        
        # Add emphasis
        emphasis_words = ['amazing', 'wonderful', 'thank', 'love', 'excited', 'beautiful', 'incredible']
        for word in emphasis_words:
            if word.lower() in text.lower():
                text = re.sub(f'\\b({word})\\b', r'<emphasis level="moderate">\1</emphasis>', 
                            text, flags=re.IGNORECASE)
        
        return text
    
    def _add_musical_phrasing(self, lyrics: str) -> str:
        """Add musical phrasing"""
        import re
        
        # Longer pauses for musical phrases
        lyrics = re.sub(r'([.!?])\s+', r'\1<break time="1200ms"/> ', lyrics)
        lyrics = re.sub(r'([,])\s+', r'\1<break time="600ms"/> ', lyrics)
        
        # Emphasize emotional words
        emotional_words = ['love', 'heart', 'soul', 'dream', 'feel', 'forever', 'always']
        for word in emotional_words:
            if word.lower() in lyrics.lower():
                lyrics = re.sub(f'\\b({word})\\b', r'<emphasis level="strong">\1</emphasis>', 
                              lyrics, flags=re.IGNORECASE)
        
        return lyrics
    
    def _convert_to_m2m(self, input_path: str, output_path: str, processing_type: str) -> bool:
        """Convert to M2M voice with simplified processing"""
        
        try:
            # Load input
            audio, sr = librosa.load(input_path, sr=self.sample_rate)
            
            # Apply M2M characteristics
            m2m_audio = self._apply_m2m_processing(audio, processing_type)
            
            # Save result
            sf.write(output_path, m2m_audio, self.sample_rate)
            
            return True
            
        except Exception as e:
            self.logger.error(f"M2M conversion error: {e}")
            return False
    
    def _apply_m2m_processing(self, audio: np.ndarray, processing_type: str) -> np.ndarray:
        """Apply M2M vocal characteristics"""
        
        # Start with input
        processed = audio.copy()
        
        # Apply pitch adjustment
        if processing_type == "singing":
            processed = librosa.effects.pitch_shift(processed, sr=self.sample_rate, n_steps=0.2)
        
        # Apply M2M formants
        processed = self._apply_formant_enhancement(processed)
        
        # Apply energy normalization
        target_rms = 0.08 if processing_type == "singing" else 0.06
        current_rms = np.sqrt(np.mean(processed**2))
        if current_rms > 0:
            processed = processed * (target_rms / current_rms)
        
        # Apply gentle compression
        processed = np.tanh(processed * 1.3) * 0.85
        
        # Add subtle character
        processed = self._add_vocal_character(processed)
        
        return processed
    
    def _apply_formant_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply M2M formant characteristics"""
        
        # Use FFT for formant enhancement
        stft = librosa.stft(audio, hop_length=256)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=stft.shape[0]*2-1)
        
        # M2M formant filter [850, 1700, 2800 Hz]
        filter_response = np.ones_like(freqs)
        
        for formant in [850, 1700, 2800]:
            # Gentle boost at formant frequencies
            boost = np.exp(-0.5 * ((freqs - formant) / (formant * 0.2))**2)
            filter_response += boost * 0.05
        
        # Apply filter
        magnitude_filtered = magnitude * filter_response[:, np.newaxis]
        stft_filtered = magnitude_filtered * np.exp(1j * phase)
        
        return librosa.istft(stft_filtered, hop_length=256)
    
    def _add_vocal_character(self, audio: np.ndarray) -> np.ndarray:
        """Add M2M vocal character"""
        
        # Subtle harmonic enhancement
        enhanced = np.tanh(audio * 1.1) * 0.95
        
        # Add minimal breath noise for naturalness
        breath_noise = np.random.normal(0, 0.001, len(audio))
        enhanced += breath_noise
        
        return enhanced

def main():
    """Demo the simple enhanced voice builder"""
    
    builder = SimpleEnhancedVoiceBuilder()
    
    print("🎭 Simple Enhanced M2M Voice Builder")
    print("=" * 50)
    print("Creating natural, non-mechanical M2M voice samples")
    print()
    
    # Test speech samples
    print("🗣️ Creating enhanced speech samples...")
    
    speech_tests = [
        {
            "text": "Thank you so much for listening to our music. It means everything to us that our voices can live on through AI.",
            "filename": "simple_speech_emotional.wav",
            "style": "emotional"
        },
        {
            "text": "Hello everyone! This is an exciting demonstration of natural AI voice synthesis. Listen to how expressive this sounds!",
            "filename": "simple_speech_excited.wav",
            "style": "excited"
        },
        {
            "text": "Walking through the digital world, where voices can be transformed, creating something beautiful and new.",
            "filename": "simple_speech_natural.wav",
            "style": "natural"
        }
    ]
    
    speech_results = []
    for test in speech_tests:
        success = builder.create_enhanced_speech(
            text=test['text'],
            output_filename=test['filename'],
            emotion_style=test['style']
        )
        speech_results.append(success)
    
    print("\n🎵 Creating enhanced singing samples...")
    
    singing_tests = [
        {
            "lyrics": "Walking through the digital world, where voices can be transformed, technology and music combined",
            "filename": "simple_singing_new_song.wav",
            "style": "melodic"
        },
        {
            "lyrics": "Mirror mirror lie to me, show me what I wanna see, in this world of dreams",
            "filename": "simple_singing_emotional.wav",
            "style": "emotional"
        },
        {
            "lyrics": "Don't say you love me, you don't even know me, if you really want me then give me some time",
            "filename": "simple_singing_ballad.wav",
            "style": "ballad"
        }
    ]
    
    singing_results = []
    for test in singing_tests:
        success = builder.create_enhanced_singing(
            lyrics=test['lyrics'],
            output_filename=test['filename'],
            emotion_style=test['style']
        )
        singing_results.append(success)
    
    # Results
    print("\n" + "=" * 50)
    print("🎉 Enhanced Voice Generation Complete!")
    print("=" * 50)
    
    successful_speech = sum(speech_results)
    successful_singing = sum(singing_results) 
    total_success = successful_speech + successful_singing
    total_tests = len(speech_results) + len(singing_results)
    
    print(f"📊 Results:")
    print(f"   🗣️ Speech: {successful_speech}/{len(speech_results)} successful")
    print(f"   🎵 Singing: {successful_singing}/{len(singing_results)} successful")
    print(f"   📈 Overall: {total_success}/{total_tests} successful ({100*total_success/total_tests:.0f}%)")
    
    if total_success > 0:
        print(f"\n📁 Generated files in: simple_enhanced_output/")
        print(f"🎧 Listen to hear the natural, non-mechanical results!")
        print(f"💡 Key improvements:")
        print(f"   • SSML emotional expression")
        print(f"   • Natural speech rhythm") 
        print(f"   • M2M formant characteristics")
        print(f"   • Reduced mechanical artifacts")
        print(f"   • Enhanced musical phrasing")
    
    print("=" * 50)

if __name__ == "__main__":
    main()