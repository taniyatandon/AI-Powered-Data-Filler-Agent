"""
Test suite for automatic Zoom & Google Meet meeting ingestion.
Verifies:
1. Zoom Cloud Recording webhook & VTT transcript parsing
2. Google Meet Space payload & export parsing
3. Automatic routing into the existing Sales Meeting Agent pipeline
4. Verified end-to-end execution without creating a second agent
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent.meeting_ingestion import (
    clean_vtt_transcript,
    parse_zoom_payload,
    parse_google_meet_payload,
    ingest_meeting,
    MeetingIngestionClient
)
from agent.sales_meeting_agent import SalesMeetingAgent


class TestMeetingIngestion(unittest.TestCase):

    def setUp(self):
        self.agent = SalesMeetingAgent(model_name="llama3.2")
        self.client = MeetingIngestionClient(agent=self.agent)

    def test_vtt_cleanup(self):
        """Tests that WebVTT/SRT timing lines and headers are cleanly stripped."""
        vtt_raw = """WEBVTT

1
00:00:01.000 --> 00:00:04.500
Alex: Hi Sarah, thanks for joining the Zoom call.

2
00:00:05.100 --> 00:00:09.200
Sarah: Hi Alex! GrowthTech needs automated marketing workflows.
"""
        cleaned = clean_vtt_transcript(vtt_raw)
        self.assertNotIn("WEBVTT", cleaned)
        self.assertNotIn("-->", cleaned)
        self.assertIn("Alex: Hi Sarah", cleaned)
        self.assertIn("Sarah: Hi Alex!", cleaned)

    def test_zoom_webhook_ingestion(self):
        """
        Tests parsing and execution of a Zoom 'recording.completed' webhook payload.
        """
        zoom_payload = {
            "event": "recording.completed",
            "payload": {
                "object": {
                    "id": "89234719283",
                    "topic": "GrowthTech Discovery & Demo Prep",
                    "duration": 25,
                    "host_email": "alex.sales@ourcompany.com",
                    "recording_files": [
                        {
                            "file_type": "TRANSCRIPT",
                            "file_extension": "VTT",
                            "transcript_content": (
                                "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\n"
                                "Alex (AE): Hi Sarah Johnson from GrowthTech.\n\n"
                                "2\n00:00:05.000 --> 00:00:09.000\n"
                                "Sarah Johnson (Marketing Director): We have a 50-person team and a $30,000 budget approved by our VP.\n\n"
                                "3\n00:00:10.000 --> 00:00:14.000\n"
                                "Sarah Johnson: HubSpot reporting was rigid. Let's do a demo Friday.\n"
                            )
                        }
                    ]
                }
            }
        }

        parsed = parse_zoom_payload(zoom_payload)
        self.assertEqual(parsed["platform"], "zoom")
        self.assertEqual(parsed["topic"], "GrowthTech Discovery & Demo Prep")
        self.assertIn("Sarah Johnson", parsed["transcript"])

        # Ingest end-to-end into agent
        ingest_res = self.client.ingest(zoom_payload, auto_confirm=True)
        self.assertTrue(ingest_res["success"])
        self.assertEqual(ingest_res["platform"], "zoom")
        
        agent_res = ingest_res["agent_result"]
        self.assertIsNotNone(agent_res.crm_data)
        self.assertEqual(agent_res.crm_data["Account"].get("Name"), "GrowthTech")
        self.assertEqual(agent_res.crm_data["Contact"].get("FirstName"), "Sarah")
        self.assertIsNotNone(agent_res.summary)
        self.assertIsNotNone(agent_res.actions.get("next_best_action"))
        self.assertIsNotNone(agent_res.crm_write_result)

        print("\n[SUCCESS] Zoom Cloud Ingestion seamlessly processed by existing Sales Meeting Agent!")

    def test_google_meet_ingestion(self):
        """
        Tests parsing and execution of a Google Meet meeting event / transcript export.
        """
        meet_payload = {
            "space": {
                "meeting_code": "abc-mnop-xyz",
                "title": "Google Meet: GrowthTech Platform Walkthrough",
            },
            "duration_minutes": 20,
            "host_email": "alex.sales@ourcompany.com",
            "transcript_content": (
                "Alex (AE): Welcome Sarah to our Google Meet.\n"
                "Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! We need automated email workflows for 50 people.\n"
                "Alex (AE): What budget is available?\n"
                "Sarah Johnson: $30,000 annually. Friday demo works for us."
            )
        }

        parsed = parse_google_meet_payload(meet_payload)
        self.assertEqual(parsed["platform"], "google_meet")
        self.assertIn("GrowthTech", parsed["topic"])

        # Ingest end-to-end
        ingest_res = ingest_meeting(meet_payload, platform="google_meet", agent=self.agent, auto_confirm=True)
        self.assertTrue(ingest_res["success"])
        self.assertEqual(ingest_res["platform"], "google_meet")
        
        agent_res = ingest_res["agent_result"]
        self.assertEqual(agent_res.crm_data["Account"].get("Name"), "GrowthTech")
        self.assertIsNotNone(agent_res.crm_write_result)

        print("\n[SUCCESS] Google Meet Ingestion seamlessly processed by existing Sales Meeting Agent!")


if __name__ == "__main__":
    unittest.main()
