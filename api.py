#!/usr/bin/env python
import os
import logging
import wave
import re
import tempfile
import shutil
import time
import uuid
from pathlib import Path
from typing import Tuple, Optional, List, Any, Dict
from dotenv import load_dotenv
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Language Mapping ---
LANGUAGE_CODE_MAP = {
    'en-US': 'a', # American English
    'en-GB': 'b', # British English
    'es-ES': 'e', # Spanish
    'fr-FR': 'f', # French
    'hi-IN': 'h', # Hindi
    'it-IT': 'i', # Italian
    'ja-JP': 'j', # Japanese
    'pt-BR': 'p', # Brazilian Portuguese
    'zh-CN': 'z', # Mandarin Chinese
}
DEFAULT_KOKORO_LANG_CODE = 'a' # Default if language not provided or not mapped

# --- Voice Mapping ---
SUPPORTED_VOICES = {
    'a': [ # American English
        'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica',
        'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah',
        'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
        'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'
    ],
    'b': [ # British English
        'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel',
        'bm_fable', 'bm_george', 'bm_lewis'
    ],
    'j': [ # Japanese
        'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'
    ],
    'z': [ # Mandarin Chinese
        'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
        'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'
    ],
    'e': [ # Spanish
        'ef_dora', 'em_alex', 'em_santa'
    ],
    'f': [ # French
        'ff_siwis'
    ],
    'h': [ # Hindi
        'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'
    ],
    'i': [ # Italian
        'if_sara', 'im_nicola'
    ],
    'p': [ # Brazilian Portuguese
        'pf_dora', 'pm_alex', 'pm_santa'
    ]
}
# Define default voices per language (e.g., the first one in the list or a known good one)
DEFAULT_VOICES = {lang: voices[0] for lang, voices in SUPPORTED_VOICES.items()}
# Optionally override specific defaults if the first isn't ideal
DEFAULT_VOICES['a'] = 'af_heart' # Explicitly set default for American English


try:
    from kokoro import KPipeline
except ImportError:
    logger.error("FATAL: Failed to import KPipeline from kokoro. Make sure 'kokoro' is installed correctly.")
    KPipeline = None 


KOKORO_SAMPLING_RATE = 24000 # Kokoro's default sampling rate

# --- Helper Functions ---
PAUSE_REGEX = re.compile(r"(\[PAUSE=(\d+(?:\.\d+)?)\])")

def ensure_dir_exists(directory_path: str):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def get_wav_duration(file_path: str) -> Optional[float]:
    """Calculates the duration of a WAV file using soundfile."""
    try:
        with sf.SoundFile(file_path) as f:
            frames = f.frames
            rate = f.samplerate
            duration = frames / float(rate) if rate > 0 else 0
            return duration
    except Exception as e:
        logger.error(f"Failed to get duration for WAV file {file_path} using soundfile: {e}")
        try:
            audio = AudioSegment.from_wav(file_path)
            duration = len(audio) / 1000.0
            logger.warning(f"Used pydub fallback for duration of {os.path.basename(file_path)}: {duration:.3f}s")
            return duration
        except Exception as pd_e:
            logger.error(f"Pydub fallback failed for {file_path}: {pd_e}")
            return None


def _generate_kokoro_segment(pipeline: KPipeline, text_segment: str, output_wav_path: str, voice_name: str) -> Optional[float]:
    """Generates a single audio segment using Kokoro with a specific voice and returns its duration."""
    cleaned_text = PAUSE_REGEX.sub('', text_segment).strip()
    if not cleaned_text:
        logger.info("Skipping empty/marker-only text segment for Kokoro.")
        return 0.0

    logger.debug(f"Sending to Kokoro: '{cleaned_text[:50]}...' (Voice: {voice_name})")

    try:
        start_time = time.time()
        # Use the provided voice_name
        generator: Any = pipeline(cleaned_text, voice=voice_name, speed=1)

        all_audio_data = []
        for _gs, _ps, audio_data in generator:
             if audio_data is not None and len(audio_data) > 0:
                 all_audio_data.append(audio_data)

        if not all_audio_data:
             logger.warning(f"Kokoro generated no audio data for segment: {cleaned_text[:50]}...")
             return 0.0

        final_audio_data = np.concatenate(all_audio_data) if len(all_audio_data) > 1 else all_audio_data[0]
        end_time = time.time()
        logger.debug(f"Kokoro segment generation took {end_time - start_time:.2f} seconds.")

        sf.write(output_wav_path, final_audio_data, KOKORO_SAMPLING_RATE)
        duration = len(final_audio_data) / float(KOKORO_SAMPLING_RATE)
        return duration

    except Exception as e:
        logger.error(f"Error during Kokoro segment generation for text: {cleaned_text[:50]}... Error: {e}", exc_info=True)
        return None

def _process_audio_generation(narration_script: str, request_id: str, target_kokoro_code: str, target_voice: str) -> Tuple[Optional[str], Optional[float], Optional[List[float]]]:
    """
    Core logic to generate audio.
    Initializes Kokoro pipeline per request based on target_kokoro_code and uses target_voice.
    Returns final path, total duration, and segment durations.
    Saves file to a temporary location.
    """
    pipeline: Optional[KPipeline] = None
    if KPipeline is None:
        logger.error(f"[{request_id}] Kokoro library (KPipeline) is not available.")
        return None, None, None
    try:
        logger.info(f"[{request_id}] Initializing Kokoro pipeline (lang_code='{target_kokoro_code}')...")
        pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code=target_kokoro_code)
        logger.info(f"[{request_id}] Kokoro pipeline initialized successfully for lang_code='{target_kokoro_code}'.")
    except Exception as e:
        logger.error(f"[{request_id}] Failed to initialize Kokoro pipeline for lang_code='{target_kokoro_code}': {e}", exc_info=True)
        return None, None, None

    request_temp_dir = tempfile.mkdtemp(prefix=f"kokoro_api_{request_id}_")
    final_output_filename = f"output_{request_id}.wav"
    final_output_filepath = os.path.join(request_temp_dir, final_output_filename)
    logger.info(f"[{request_id}] Generating audio with Kokoro TTS...")

    parts = PAUSE_REGEX.split(narration_script)
    parts = [p for p in parts if p]

    combined_audio = AudioSegment.empty()
    segment_durations: List[float] = []
    temp_segment_dir = os.path.join(request_temp_dir, "segments")
    ensure_dir_exists(temp_segment_dir)
    logger.debug(f"[{request_id}] Using temp segment directory: {temp_segment_dir}")
    success = True

    try:
        segment_index = 0
        for i, part in enumerate(parts):
            match = PAUSE_REGEX.match(part)
            if match:
                pause_duration_str = match.group(2)
                try:
                    pause_duration_ms = int(float(pause_duration_str) * 1000)
                    if pause_duration_ms > 0:
                        logger.debug(f"[{request_id}] Adding {pause_duration_ms}ms silence.")
                        combined_audio += AudioSegment.silent(duration=pause_duration_ms)
                except ValueError:
                    logger.warning(f"[{request_id}] Invalid pause duration format: {part}. Ignoring pause.")
            else:
                text_segment = part.strip()
                if not text_segment: continue

                segment_filename = os.path.join(temp_segment_dir, f"segment_{segment_index}.wav")
                # Pass the target_voice to the segment generation function
                segment_duration = _generate_kokoro_segment(pipeline, text_segment, segment_filename, target_voice)

                if segment_duration is not None:
                    if segment_duration > 0.01:
                        try:
                            audio_segment = AudioSegment.from_wav(segment_filename)
                            combined_audio += audio_segment
                            segment_durations.append(segment_duration)
                            segment_index += 1
                        except Exception as e:
                             logger.error(f"[{request_id}] Failed to load generated segment {segment_filename}: {e}", exc_info=True)
                             success = False; break
                    else:
                         logger.info(f"[{request_id}] Skipping segment {segment_index} due to zero duration.")
                else:
                    logger.error(f"[{request_id}] Failed to generate audio for segment: {text_segment[:50]}...")
                    success = False; break

        if not success or len(combined_audio) == 0:
             logger.error(f"[{request_id}] Audio generation failed or resulted in empty audio.")
             # Cleanup happens in the finally block
             return None, None, None

        logger.info(f"[{request_id}] Exporting combined audio ({len(combined_audio) / 1000.0:.2f}s) to {final_output_filepath}")
        combined_audio.export(final_output_filepath, format="wav")

        total_duration = get_wav_duration(final_output_filepath)
        if total_duration is None:
             logger.warning(f"[{request_id}] Could not get duration from final file, using pydub length.")
             total_duration = len(combined_audio) / 1000.0

        logger.info(f"[{request_id}] Successfully generated final audio (Duration: {total_duration:.2f}s).")
        logger.debug(f"[{request_id}] Speech segment durations: {segment_durations}")
        # Return the path to the final file within its unique temp dir
        return final_output_filepath, total_duration, segment_durations

    except Exception as e:
         logger.error(f"[{request_id}] An unexpected error occurred during audio processing: {e}", exc_info=True)
         return None, None, None

def cleanup_temp_dir(temp_dir_path: str):
    """Safely removes a temporary directory."""
    try:
        if os.path.exists(temp_dir_path):
            shutil.rmtree(temp_dir_path)
            logger.info(f"Cleaned up temporary directory: {temp_dir_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory {temp_dir_path}: {e}", exc_info=True)


router = APIRouter()

@router.post("/generate_audio/",
          response_class=FileResponse,
          responses={
              200: {
                  "content": {"audio/wav": {}},
                  "description": "Successful audio generation. Returns WAV file.",
              },
              422: {"description": "Validation Error (e.g., missing text or invalid language)"},
              500: {"description": "Internal Server Error (e.g., TTS failure or pipeline init failure)"},
          })
async def generate_audio_endpoint(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...)
    ):
    """
    Generates audio from the provided text using Kokoro TTS, selecting language based on payload.

    - **payload**: JSON body containing:
        - **text** (str): The narration script, potentially with [PAUSE=...] markers.
        - **language** (str, optional): The desired language code (e.g., "en-US", "hi-IN"). Defaults to American English if omitted or invalid.
        - **voice** (str, optional): The desired voice name (e.g., "af_heart", "bf_emma"). Defaults to a language-specific default if omitted or invalid.
    """

    narration_script = payload.get("text")
    if not narration_script or not isinstance(narration_script, str):
        raise HTTPException(status_code=422, detail="Missing or invalid 'text' field in request body.")

    # Get language from payload and map to Kokoro code
    requested_language = payload.get("language")
    target_kokoro_code = LANGUAGE_CODE_MAP.get(requested_language, DEFAULT_KOKORO_LANG_CODE) if requested_language else DEFAULT_KOKORO_LANG_CODE
    logger.info(f"Requested language: '{requested_language}', Mapped Kokoro code: '{target_kokoro_code}'")

    # Determine the target voice
    requested_voice = payload.get("voice")
    target_voice = DEFAULT_VOICES.get(target_kokoro_code, list(SUPPORTED_VOICES.values())[0][0]) # Fallback to absolute first voice if lang somehow not in DEFAULT_VOICES

    if requested_voice:
        if target_kokoro_code in SUPPORTED_VOICES and requested_voice in SUPPORTED_VOICES[target_kokoro_code]:
            target_voice = requested_voice
            logger.info(f"Using requested voice: '{target_voice}' for language code '{target_kokoro_code}'")
        else:
            logger.warning(f"Requested voice '{requested_voice}' is not valid for language code '{target_kokoro_code}'. Falling back to default: '{target_voice}'")
    else:
        logger.info(f"No voice requested, using default for language code '{target_kokoro_code}': '{target_voice}'")


    request_id = str(uuid.uuid4())
    logger.info(f"Received audio generation request {request_id} for language '{requested_language or 'default'}' and voice '{target_voice}'")

    temp_dir_path = None
    try:
        temp_dir_path = os.path.join(tempfile.gettempdir(), f"kokoro_api_{request_id}_")
        final_output_filepath, total_duration, segment_durations = _process_audio_generation(
            narration_script, request_id, target_kokoro_code, target_voice
        )

        if final_output_filepath and os.path.exists(final_output_filepath):
            logger.info(f"[{request_id}] Sending audio file: {final_output_filepath}")
            background_tasks.add_task(cleanup_temp_dir, os.path.dirname(final_output_filepath))
            return FileResponse(
                path=final_output_filepath,
                media_type='audio/wav',
                filename=f"generated_audio_{request_id}.wav"
            )
        else:
            logger.error(f"[{request_id}] Audio generation process failed to produce a file.")
            if temp_dir_path:
                 background_tasks.add_task(cleanup_temp_dir, temp_dir_path)
            raise HTTPException(status_code=500, detail="Audio generation failed.")

    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception in /generate_audio endpoint: {e}", exc_info=True)
        if temp_dir_path:
            background_tasks.add_task(cleanup_temp_dir, temp_dir_path)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/health")
async def health_check():
    """Basic health check endpoint. Checks if Kokoro library is importable."""
    if KPipeline is not None:
         return {"status": "ok", "message": "Kokoro library (KPipeline) is available. Initialization happens per request."}
    else:
         return {"status": "error", "message": "Kokoro library (KPipeline) is not available/importable."}
