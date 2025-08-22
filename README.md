# Kokoro TTS Audio API (FastAPI)

Turn text into speech (WAV). Built with FastAPI and the kokoro TTS library.

- Input: text (supports pauses like `[PAUSE=1.5]`)
- Output: audio/wav

## Prerequisites

- Python 3.8+
- pip
- System libs for Japanese tokenization (needed by kokoro):
  - Ubuntu: `sudo apt install mecab libmecab-dev mecab-ipadic-utf8 cmake`
  - Arch: `yay -S mecab-git && sudo pacman -S cmake`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python3 -m unidic download
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- API docs: http://localhost:8000/docs

## API

### POST /api/generate_audio/
- Body (JSON):
  - text: string (required). Example: `"Hello. [PAUSE=1] Next sentence."`
  - language: string (optional). Default: `en-US` (examples: `en-GB`, `ja-JP`, `hi-IN`).
  - voice: string (optional). If invalid/missing, a default for the language is used.
- Response: `audio/wav` (file stream)

Example:
```bash
curl -X POST "http://localhost:8000/api/generate_audio/" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hello world. [PAUSE=1.5] This is a test.",
           "language": "en-US",
           "voice": "af_heart"
         }' \
     --output english.wav
```

Japanese example:
```bash
curl -X POST "http://localhost:8000/api/generate_audio/" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "こんにちは、世界！ [PAUSE=1] これはテストです。",
           "language": "ja-JP",
           "voice": "jf_nezumi"
         }' \
     --output japanese.wav
```

Tips:
- Use `[PAUSE=seconds]` to add silence between sentences.
- Supported languages and voices are defined in `api.py` (`LANGUAGE_CODE_MAP`, `SUPPORTED_VOICES`).