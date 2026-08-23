"""
Audio Transcription & Speech-to-Text Module for Sales Meeting Recordings.
Uses Whisper to transcribe uploaded meeting recordings (.wav, .mp3, .m4a, .ogg, .webm, .flac)
and feeds the output transcript directly into the Sales Meeting Agent.
"""

import io
import os
import wave
import tempfile
from pathlib import Path
from typing import Dict, Any, Union, BinaryIO, Optional
import numpy as np

_WHISPER_MODEL_CACHE: Dict[str, Any] = {}


def get_whisper_model(model_size: str = "base"):
    """
    Loads and caches the Whisper speech-to-text model.
    Default to 'base' for fast, high-quality transcription on CPU/GPU.
    """
    global _WHISPER_MODEL_CACHE
    if model_size not in _WHISPER_MODEL_CACHE:
        try:
            import whisper
            _WHISPER_MODEL_CACHE[model_size] = whisper.load_model(model_size)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{model_size}': {e}")
    return _WHISPER_MODEL_CACHE[model_size]


def _load_audio_to_numpy(audio_source: Union[str, Path, bytes, BinaryIO, np.ndarray]) -> np.ndarray:
    """
    Decodes audio from numpy arrays, file paths, raw bytes, or stream buffers into a 16kHz mono float32 numpy array.
    Uses soundfile or wave to avoid external ffmpeg requirements.
    """
    target_sr = 16000

    # 1. Direct numpy array input
    if isinstance(audio_source, np.ndarray):
        audio = audio_source.copy()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                max_val = np.iinfo(audio.dtype).max
                audio = (audio / max_val).astype(np.float32)
            else:
                audio = audio.astype(np.float32)
        return audio

    # Extract raw bytes if stream or bytes-like object
    raw_bytes = None
    if isinstance(audio_source, (bytes, bytearray)):
        raw_bytes = bytes(audio_source)
    elif hasattr(audio_source, "read"):
        raw_bytes = audio_source.read()
        if hasattr(audio_source, "seek"):
            try:
                audio_source.seek(0)
            except Exception:
                pass

    # 2. Try soundfile first (supports WAV, FLAC, OGG, etc.)
    try:
        import soundfile as sf
        if raw_bytes is not None:
            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        else:
            audio, sr = sf.read(str(audio_source), dtype="float32")

        # Convert stereo/multi-channel to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != target_sr:
            indices = np.round(np.arange(0, len(audio), sr / target_sr)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        return audio.astype(np.float32)
    except Exception:
        pass

    # 3. Try standard library wave module for WAV files/bytes
    try:
        if raw_bytes is not None:
            wav_stream = io.BytesIO(raw_bytes)
        else:
            wav_stream = open(str(audio_source), "rb")

        with wave.open(wav_stream, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)

            if sampwidth == 2:
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            if framerate != target_sr:
                indices = np.round(np.arange(0, len(audio), framerate / target_sr)).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

            return audio.astype(np.float32)
    except Exception:
        pass

    # 4. Try Whisper's native loader via temp file if raw bytes are available
    if raw_bytes is not None:
        try:
            import whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                audio = whisper.load_audio(tmp_path)
                return audio.astype(np.float32)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception:
            pass

    raise ValueError("Could not decode audio format. Please provide a standard .wav, .flac, .ogg, or .mp3 file.")



def transcribe_meeting_audio(
    audio_source: Union[str, Path, bytes, BinaryIO],
    model_size: str = "base",
    language: Optional[str] = "en"
) -> Dict[str, Any]:
    """
    Transcribes a sales meeting audio recording into formatted text.
    
    Args:
        audio_source: File path (str/Path), raw bytes, or file-like buffer (from Streamlit uploader).
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium').
        language: Language code (default 'en' for English).
        
    Returns:
        Dict containing:
        - transcript: str (the full transcribed dialogue text)
        - language: str (detected or specified language)
        - segments: list of segment objects with timestamps
    """
    try:
        # 1. Load and decode audio to 16kHz float32 numpy array
        audio_np = _load_audio_to_numpy(audio_source)

        # 2. Run Whisper speech-to-text transcription
        model = get_whisper_model(model_size=model_size)
        options = {"fp16": False}  # CPU safe
        if language:
            options["language"] = language

        result = model.transcribe(audio_np, **options)
        transcript_text = (result.get("text") or "").strip()

        # Format segments if present
        segments = result.get("segments", [])

        return {
            "transcript": transcript_text,
            "language": result.get("language", language or "en"),
            "segments": segments,
            "success": True,
        }

    except Exception as e:
        return {
            "transcript": "",
            "language": language or "en",
            "segments": [],
            "success": False,
            "error": str(e),
        }
