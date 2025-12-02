# Local Voice Bot (Phase 2)

A fully local, autonomous voice assistant using Pipecat, Faster-Whisper, Ollama, and Silero TTS.

## Prerequisites

- **Hardware**: NVIDIA GPU (RTX 3060 or better recommended) with CUDA support.
- **Software**:
  - Python 3.10+
  - [Ollama](https://ollama.com/) installed and running.
  - [Daily.co](https://daily.co/) account (for WebRTC transport).

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may need to install PyTorch with CUDA support manually if the default installation doesn't detect your GPU correctly. See [pytorch.org](https://pytorch.org/get-started/locally/).*

2. **Configure Ollama**:
   Pull the Llama 3 model (or your preferred model):
   ```bash
   ollama pull llama3
   ```
   Ensure Ollama is running:
   ```bash
   ollama serve
   ```

3. **Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   DAILY_ROOM_URL=https://your-domain.daily.co/your-room
   DAILY_TOKEN=your-daily-token
   ```

## Running the Bot

You can use the helper script:
```bash
./run.sh
```

Or run directly with Python:
```bash
python3 bot.py
```

## Architecture

- **STT**: Faster-Whisper (CTranslate2) running on GPU.
- **LLM**: Llama 3 via Ollama (Localhost API).
- **TTS**: Silero V5 (Russian) running on CPU/GPU.
- **Transport**: Daily.co WebRTC.
