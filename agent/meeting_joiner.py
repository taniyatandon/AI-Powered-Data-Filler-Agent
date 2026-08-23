"""
Automated Meeting Joiner & Live Meeting Stream Capture.
Joins live Google Meet, Zoom, or web meetings as an automated listener participant.
Captures live audio / captions, streams the live conversation transcript,
and automatically triggers the Sales Meeting Agent for summary and CRM extraction upon meeting completion.
"""

import re
import time
import queue
import threading
from typing import Optional, Callable, Dict, Any, List

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult


class GoogleMeetJoiner:
    """
    Automates joining a Google Meet URL, turns off camera/mic,
    activates live captions, captures dialogue stream, and auto-summarizes.
    """

    def __init__(
        self,
        participant_name: str = "AI Sales Agent",
        agent: Optional[SalesMeetingAgent] = None,
        on_segment_callback: Optional[Callable[[str], None]] = None
    ):
        self.participant_name = participant_name
        self.agent = agent or SalesMeetingAgent()
        self.on_segment_callback = on_segment_callback

        self.driver = None
        self._is_running = False
        self._listener_thread = None
        self.captured_transcript_lines: List[str] = []
        self._seen_texts = set()
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def join_meeting(self, meet_url: str, headless: bool = False) -> bool:
        """
        Launches browser and joins the Google Meet room.
        """
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is required for automated meeting joining. Please ensure selenium is installed.")

        # Validate URL
        if not meet_url or "meet.google.com" not in meet_url:
            if not meet_url.startswith("http"):
                meet_url = f"https://meet.google.com/{meet_url.strip()}"

        chrome_options = ChromeOptions()
        chrome_options.add_argument("--use-fake-ui-for-media-stream")
        chrome_options.add_argument("--use-fake-device-for-media-stream")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--start-maximized")
        if headless:
            chrome_options.add_argument("--headless=new")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            # Fallback to standard Chrome init
            try:
                self.driver = webdriver.Chrome()
            except Exception as e2:
                raise RuntimeError(f"Could not initialize Chrome browser: {e2}")

        try:
            self.driver.get(meet_url)
            self._is_running = True
            time.sleep(3.0)

            # 1. Fill display name if prompted
            try:
                name_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@type='text' or @aria-label='Your name' or @placeholder='Your name']"))
                )
                name_input.clear()
                name_input.send_keys(self.participant_name)
                time.sleep(0.5)
            except Exception:
                pass

            # 2. Turn off mic and camera (Ctrl+d, Ctrl+e)
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.CONTROL + "d")  # Mute mic
                time.sleep(0.5)
                body.send_keys(Keys.CONTROL + "e")  # Turn off cam
                time.sleep(0.5)
            except Exception:
                pass

            # 3. Click Ask to join / Join now button
            try:
                join_button = WebDriverWait(self.driver, 8).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Ask to join') or contains(., 'Join now') or contains(., 'Join')]"))
                )
                join_button.click()
            except Exception:
                pass

            # 4. Turn on captions (hotkey 'c' in Google Meet)
            time.sleep(3.0)
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.send_keys("c")
            except Exception:
                pass

            # 5. Start background caption scraper thread
            self._listener_thread = threading.Thread(target=self._scrape_captions_loop, daemon=True)
            self._listener_thread.start()

            return True

        except Exception as e:
            self._is_running = False
            if self.driver:
                try:
                    self.driver.quit()
                except Exception:
                    pass
                self.driver = None
            raise RuntimeError(f"Failed to join Google Meet room: {e}")

    def _scrape_captions_loop(self):
        """Scrapes real-time captions from Google Meet UI."""
        while self._is_running and self.driver:
            try:
                # Google Meet caption text selectors
                captions = self.driver.find_elements(By.CSS_SELECTOR, "div[jsname='YSxPfc'], div.yg51vc, span.VfPpkd-vQzf8d")
                for cap in captions:
                    txt = cap.text.strip()
                    if txt and len(txt) > 3 and txt not in self._seen_texts:
                        with self._lock:
                            self._seen_texts.add(txt)
                            self.captured_transcript_lines.append(txt)
                        if self.on_segment_callback:
                            try:
                                self.on_segment_callback(txt)
                            except Exception:
                                pass
                time.sleep(1.0)
            except Exception:
                time.sleep(1.0)

    def get_accumulated_transcript(self) -> str:
        """Returns the full transcript scraped so far."""
        with self._lock:
            return "\n".join(self.captured_transcript_lines)

    def leave_and_summarize(self, fallback_sample_if_empty: str = "") -> AgentResult:
        """
        Leaves the meeting, closes the browser, and routes the transcript
        directly to the Sales Meeting Agent for the 9-point summary, CRM sync, and NBA.
        """
        self._is_running = False
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)

        transcript = self.get_accumulated_transcript().strip()
        if not transcript:
            transcript = fallback_sample_if_empty.strip()

        if not transcript:
            raise ValueError("Meeting transcript is empty. No speech was detected during the Google Meet call.")

        return self.agent.run_end_to_end_pipeline(transcript, auto_confirm=True)
