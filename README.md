# Higgs Audio v2

> An advanced end-to-end multimodal model for understanding and generating high-quality audio from text descriptions with support for voice cloning, multiple speakers, and diverse styles.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Programmatic API](#programmatic-api)
- [Model Capabilities](#model-capabilities)
- [Voice Presets](#voice-presets)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

Higgs Audio v2 is a state-of-the-art multimodal model that generates natural-sounding audio from text descriptions. It combines advanced text-to-speech technology with support for voice cloning, multi-speaker dialogue, and contextual audio generation. The model is built on transformer-based architectures and leverages the Descript Audio Codec (DAC) for high-fidelity audio encoding and decoding.

**Key Stats:**

- **Model Size:** 3B parameters (generation base model)
- **Audio Codec:** Descript Audio Codec (DAC)
- **Sample Rate:** 24 kHz
- **Supported Languages:** English, Chinese (and more)
- **GPU Support:** CUDA-optimized inference

---

## Features

### Core Capabilities

✨ **Text-to-Speech Generation**

- Convert text descriptions directly to high-quality audio
- Contextual audio generation with scene descriptions
- Multi-language support (English, Chinese, etc.)

🎤 **Voice Cloning**

- Clone reference voices from user-provided audio samples
- Transfer voice characteristics while preserving text content
- Support for custom voice presets

👥 **Multi-Speaker Support**

- Generate dialogues with multiple distinct speakers
- Control speaker characteristics through descriptions
- Support for different voice personalities and accents

🎵 **Audio Enhancement Features**

- Background music (BGM) support with `[music start]` and `[music end]` tags
- Scene descriptions for controlling recording quality and environment
- Fine-grained speaker control through descriptive prompts

🌍 **Multilingual**

- Full English support
- Chinese language support with proper punctuation handling
- Extensible to additional languages

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU fallback
- 8GB+ VRAM for optimal inference speed

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/higgs_audio_v2.git
cd higgs_audio_v2
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or install from `pyproject.toml`:

```bash
pip install -e .
```

### Dependencies

Key dependencies include:

- `torch>=2.5.1` - Deep learning framework
- `torchaudio>=2.5.1` - Audio processing
- `transformers>=4.45.1` - Transformer models
- `gradio` - Web UI framework
- `descript-audio-codec` - Audio encoding/decoding
- `librosa` - Audio analysis
- `loguru` - Logging

See `requirements.txt` for the complete list.

---

## Quick Start

### Web Interface (Recommended)

1. **Start the Gradio app:**

```bash
python app.py
```

2. **Open your browser** and navigate to `http://localhost:7860`

3. **Choose a preset example** or create your own:
   - **Voice Clone:** Upload a reference audio sample
   - **Smart Voice:** Generate audio from text description
   - **Multi-Speaker:** Create dialogue with multiple speakers
   - **Voice Description:** Control voices with detailed descriptions

### Programmatic API

```python
from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
from higgs_audio.data_types import ChatMLSample, Message

# Initialize the engine
engine = HiggsAudioServeEngine(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
    device="cuda"
)

# Create a simple chat sample
sample = ChatMLSample(
    messages=[
        Message(role="system", content="Generate audio following instruction."),
        Message(role="user", content="Hello, this is a test message.")
    ]
)

# Generate audio
audio_data, sample_rate = engine.generate(sample)

# Save the audio
import scipy.io.wavfile as wavfile
wavfile.write("output.wav", sample_rate, audio_data)
```

---

## Usage

### Web Interface Guide

#### 1. Voice Clone

Upload a reference audio file to clone its voice characteristics.

**Example:**

- Upload audio: `reference_voice.wav`
- Input text: "I'm learning a new language"
- Result: Audio with the voice characteristics from the reference

#### 2. Smart Voice

Generate audio based on a system prompt and context.

**Example System Prompt:**

```
Generate audio following instruction.

<|scene_desc_start|>
Audio is recorded from a quiet room.
<|scene_desc_end|>
```

#### 3. Multi-Speaker Voice Description

Create dialogues with multiple distinct speakers.

**Example System Prompt:**

```
<|scene_desc_start|>
SPEAKER0: feminine
SPEAKER1: masculine
<|scene_desc_end|>
```

**Input Text:**

```
[SPEAKER0] Hello, how are you today?
[SPEAKER1] I'm doing great, thanks for asking!
```

#### 4. Single-Speaker with Voice Description

Control specific vocal characteristics.

**Example System Prompt:**

```
<|scene_desc_start|>
SPEAKER0: She has a warm, friendly voice with a slight accent. Speaks at a moderate pace.
<|scene_desc_end|>
```

#### 5. Multi-Language Support

- Switch between English and Chinese
- Automatic punctuation normalization for Chinese text

#### 6. Background Music (Experimental)

Add background music using `[music start]` and `[music end]` tags.

```
[music start] This is some ambient music [music end] while I continue speaking normally.
```

---

## Model Capabilities

### Architecture

The model consists of:

1. **Audio Encoder** - Converts audio input to features (for voice cloning)
2. **Text/Multimodal Processor** - Processes text and audio inputs
3. **Transformer Decoder** - Generates audio tokens autoregressively
4. **Audio Decoder (DAC)** - Converts tokens back to waveform

### Generation Modes

- **Text-to-Speech** - Pure text input to audio
- **Voice-Cloning** - Reference audio + text → audio in reference voice
- **Dialogue** - Multi-speaker text with voice descriptions
- **Conditional Generation** - Scene descriptions and detailed voice prompts

### Fine-Tuning Support

The codebase supports fine-tuning through:

- Custom `ChatMLSample` datasets
- Distributed training (DDP) support
- CUDA graph optimization for inference

---

## Voice Presets

The project includes pre-configured voice presets for quick testing. Configure them in `voice_examples/config.json`:

```json
{
  "preset_name": {
    "transcript": "Sample text for this voice"
  }
}
```

Add corresponding `.wav` files in the `voice_examples/` directory.

---

## Configuration

### Model Configuration

Modify settings in `higgs_audio/model/configuration_higgs_audio.py`:

- **Model size**: Adjust attention heads, hidden dimensions
- **Audio processing**: Sample rate, codec parameters
- **Quantization**: Vector quantization settings

### Inference Configuration

Key parameters in the Gradio app (`app.py`):

```python
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
SAMPLE_RATE = 24000
```

### System Prompts

Customize the `DEFAULT_SYSTEM_PROMPT` in `app.py` to change default behavior.

---

## Architecture

### Directory Structure

```
higgs_audio/
├── audio_processing/          # Audio encoding/decoding
│   ├── higgs_audio_tokenizer.py
│   ├── semantic_module.py
│   ├── descriptaudiocodec/     # DAC implementation
│   └── quantization/           # Vector quantization
├── data_collator/              # Data batching utilities
├── dataset/                    # Dataset handling
│   └── chatml_dataset.py       # ChatML format support
├── model/                      # Core model architecture
│   ├── modeling_higgs_audio.py
│   ├── audio_head.py
│   ├── configuration_higgs_audio.py
│   └── custom_modules.py       # Specialized layers
├── serve/                      # Inference engine
│   └── serve_engine.py
└── constants.py                # Shared constants
```

### Key Components

| Component                  | Purpose           |
| -------------------------- | ----------------- |
| `HiggsAudioModel`          | Main model class  |
| `HiggsAudioServeEngine`    | Inference wrapper |
| `HiggsAudioSampleCollator` | Batch processing  |
| `DAC`                      | Audio codec       |
| `ChatMLSample`             | Data format       |

---

## Requirements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+ system RAM for smooth operation
- **Storage**: 20GB+ for model weights

### Python Packages

See `requirements.txt` for full list. Key packages:

```
torch==2.5.1
torchaudio==2.5.1
transformers>=4.45.1
descript-audio-codec
librosa
gradio
```

---

## Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce batch size or use CPU mode:

```python
engine = HiggsAudioServeEngine(device="cpu")
```

### Audio Quality Issues

**Solution:** Adjust model parameters or try different system prompts:

- Use detailed scene descriptions
- Provide high-quality reference audio for voice cloning
- Adjust voice descriptions for better control

### Import Errors

**Solution:** Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

### Model Download Issues

**Solution:** Pre-download models from Hugging Face:

```bash
huggingface-cli download bosonai/higgs-audio-v2-generation-3B-base
huggingface-cli download bosonai/higgs-audio-v2-tokenizer
```

---

## Citation

If you use Higgs Audio in your research, please cite:

```bibtex
@misc{higgs_audio_v2,
  title={Higgs Audio v2: Advanced Multimodal Audio Generation},
  year={2025},
  note={Boson AI}
}
```

---

## License

See LICENSE file for details.

---

## Support & Community

For questions, issues, or contributions:

- Open an issue on GitHub
- Check existing documentation
- Review example notebooks in `voice_examples/`

---

**Happy audio generation! 🎵**
