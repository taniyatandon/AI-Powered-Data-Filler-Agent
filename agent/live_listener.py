"""
Live Meeting Listener & Real-Time Audio Transcription Engine.
Listens to live meeting conversations directly from the microphone / system audio stream
(like a human taking notes in real time, without requiring third-party bots).
Continuously captures live speech, transcribes segments using Whisper,
and automatically generates comprehensive meeting summaries and CRM actions when the meeting concludes.
"""

import time
import queue
import threading
import numpy as np
from typing import Optional, Callable, List, Dict, Any

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

from agent.transcriber import transcribe_meeting_audio
from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult


class LiveMeetingListener:
    """
    Listens to live sales meetings in real time, generates streaming transcripts,
    and automatically forwards the complete meeting transcript to the Sales Meeting Agent.
    """

    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = 16000,
        chunk_duration_sec: float = 4.0,
        energy_threshold: float = 0.008,
        on_segment_callback: Optional[Callable[[str], None]] = None,
        agent: Optional[SalesMeetingAgent] = None,
    ):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.energy_threshold = energy_threshold
        self.on_segment_callback = on_segment_callback
        self.agent = agent or SalesMeetingAgent()

        self._is_listening = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._transcribe_thread: Optional[threading.Thread] = None
        self._audio_stream: Optional[Any] = None

        self.transcript_segments: List[Dict[str, Any]] = []
        self.accumulated_transcript: str = ""
        self._lock = threading.Lock()
        self._start_time: float = 0.0

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback invoked by sounddevice for incoming audio buffers."""
        if not self._is_listening:
            return
        # Convert to 1D float32 mono
        if indata.ndim > 1:
            mono_data = indata.mean(axis=1)
        else:
            mono_data = indata.flatten()
        self._audio_queue.put(mono_data.copy().astype(np.float32))

    def _transcription_worker(self):
        """Worker thread that processes accumulated audio chunks and runs Whisper."""
        buffer = []
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_sec)

        while self._is_listening or not self._audio_queue.empty():
            try:
                data = self._audio_queue.get(timeout=0.25)
                buffer.append(data)
            except queue.Empty:
                continue

            current_samples = sum(len(b) for b in buffer)
            if current_samples >= samples_per_chunk or (not self._is_listening and buffer):
                full_chunk = np.concatenate(buffer)
                buffer = []

                # Compute RMS energy for voice activity
                rms = np.sqrt(np.mean(full_chunk ** 2))
                if rms < self.energy_threshold and self._is_listening:
                    # Skip background silence / ambient noise
                    continue

                # Transcribe speech chunk
                try:
                    res = transcribe_meeting_audio(full_chunk, model_size=self.model_size)
                    text = (res.get("transcript") or "").strip()
                    if text and len(text) > 2:
                        elapsed_sec = int(time.time() - self._start_time)
                        mins, secs = divmod(elapsed_sec, 60)
                        timestamp = f"{mins:02d}:{secs:02d}"

                        segment_info = {
                            "timestamp": timestamp,
                            "text": text,
                        }

                        with self._lock:
                            self.transcript_segments.append(segment_info)
                            if self.accumulated_transcript:
                                self.accumulated_transcript += f"\n[{timestamp}] {text}"
                            else:
                                self.accumulated_transcript = f"[{timestamp}] {text}"

                        if self.on_segment_callback:
                            try:
                                self.on_segment_callback(f"[{timestamp}] {text}")
                            except Exception:
                                pass
                except Exception as e:
                    # Non-fatal chunk error
                    pass

    def start(self, device: Optional[int] = None) -> bool:
        """
        Starts listening to live audio from the microphone / default audio device.
        """
        if self._is_listening:
            return True

        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("Live audio recording requires 'sounddevice'. Please ensure sounddevice is installed.")

        self._is_listening = True
        self._start_time = time.time()
        self.transcript_segments = []
        self.accumulated_transcript = ""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start transcription worker thread
        self._transcribe_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._transcribe_thread.start()

        # Start sounddevice InputStream
        try:
            self._audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.5)
            )
            self._audio_stream.start()
            return True
        except Exception as e:
            self._is_listening = False
            if self._transcribe_thread and self._transcribe_thread.is_alive():
                self._transcribe_thread.join(timeout=1.0)
            raise RuntimeError(f"Could not open live microphone stream: {e}")

    def stop(self, wait_for_drain: bool = True) -> str:
        """
        Stops listening and returns the complete accumulated transcript.
        """
        if not self._is_listening:
            return self.accumulated_transcript

        self._is_listening = False

        if self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        if wait_for_drain and self._transcribe_thread:
            self._transcribe_thread.join(timeout=5.0)

        return self.accumulated_transcript

    def feed_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Feeds an external audio buffer into the live listener (used for testing or streaming audio feeds).
        """
        if audio_data.ndim > 1:
            mono = audio_data.mean(axis=1)
        else:
            mono = audio_data.flatten()
        self._audio_queue.put(mono.astype(np.float32))

    def get_live_transcript(self) -> str:
        """Returns the transcript accumulated so far during the live call."""
        with self._lock:
            return self.accumulated_transcript

    def process_and_summarize(self, auto_confirm: bool = True) -> AgentResult:
        """
        Finalizes the live listening session and automatically routes the accumulated transcript
        into the complete 8-stage Sales Meeting Agent pipeline (automatic 9-point summary,
        CRM update, coaching, Next Best Action, and action items).
        """
        transcript = self.stop()
        if not transcript.strip():
            raise ValueError("Live meeting transcript is empty. No speech was detected during the session.")

        return self.agent.run_end_to_end_pipeline(transcript, auto_confirm=auto_confirm)


def listen_to_live_meeting_and_summarize(
    duration_seconds: float = 10.0,
    agent: Optional[SalesMeetingAgent] = None,
    auto_confirm: bool = True
) -> AgentResult:
    """
    Convenience function that listens to live meeting audio for a specified duration
    and automatically produces the complete meeting summary and CRM actions.
    """
    listener = LiveMeetingListener(agent=agent)
    listener.start()
    time.sleep(duration_seconds)
    return listener.process_and_summarize(auto_confirm=auto_confirm)
