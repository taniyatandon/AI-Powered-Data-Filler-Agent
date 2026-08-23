"""
Agent package initialization.
"""

from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.summarizer import summarize_meeting_transcript
from agent.coaching import generate_meeting_coaching
from agent.next_best_action import determine_next_best_action
from agent.transcriber import transcribe_meeting_audio
from agent.extractor_ollama import extract_crm_data as extract_crm_ollama
from agent.validator import validate_extraction, ValidationError
from agent.mock_crm_client import MockCRMClient

from agent.meeting_ingestion import (
    ingest_meeting,
    MeetingIngestionClient,
    clean_vtt_transcript,
    parse_zoom_payload,
    parse_google_meet_payload
)
from agent.live_listener import (
    LiveMeetingListener,
    listen_to_live_meeting_and_summarize
)
from agent.meeting_joiner import GoogleMeetJoiner

__all__ = [
    "SalesMeetingAgent",
    "AgentResult",
    "summarize_meeting_transcript",
    "generate_meeting_coaching",
    "determine_next_best_action",
    "transcribe_meeting_audio",
    "extract_crm_ollama",
    "validate_extraction",
    "ValidationError",
    "MockCRMClient",
    "ingest_meeting",
    "MeetingIngestionClient",
    "clean_vtt_transcript",
    "parse_zoom_payload",
    "parse_google_meet_payload",
    "LiveMeetingListener",
    "listen_to_live_meeting_and_summarize",
    "GoogleMeetJoiner",
]



