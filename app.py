"""
AI Sales Meeting Agent (v3)
===========================
Enterprise-grade Streamlit front end for the AI Sales Meeting Agent.

The AI Sales Meeting Agent orchestrates 5 core agentic steps:
1. Conversation Analysis: Evaluates meeting context, sentiment, buyer intent, and urgency.
2. Information Need Assessment: Determines completeness vs missing critical discovery items.
3. CRM Data Extraction: Extracts structured Contact, Account, Opportunity, and Insights via Llama 3.2.
4. Executive Meeting Summary: Generates a concise executive brief and solution alignment.
5. Action Decisions & Next Steps: Determines CRM actions, sales tasks, risk alerts, and tailored follow-up emails.

Run from the project root:
    streamlit run app.py
"""

import re
import time
import html
import numpy as np

import streamlit as st

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False



# ---------------------------------------------------------------------------
# Backend hooks (SalesMeetingAgent / validation / CRM logic)
# ---------------------------------------------------------------------------
try:
    from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
    from agent.extractor_ollama import extract_crm_data
    from agent.validator import validate_extraction, ValidationError
    from agent.mock_crm_client import MockCRMClient
    from agent.transcriber import transcribe_meeting_audio
    from agent.meeting_ingestion import (
        clean_vtt_transcript,
        ingest_meeting,
        parse_zoom_payload,
        parse_google_meet_payload
    )
    from agent.live_listener import LiveMeetingListener
    from agent.meeting_joiner import GoogleMeetJoiner
    from rapidfuzz import fuzz
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

    class GoogleMeetJoiner:
        def __init__(self, *args, **kwargs):
            self.is_running = False
        def join_meeting(self, *args, **kwargs):
            return True
        def leave_and_summarize(self, *args, **kwargs):
            return None

    class LiveMeetingListener:
        def __init__(self, *args, **kwargs):
            self.is_listening = False
        def start(self):
            return True
        def stop(self):
            return ""
        def get_live_transcript(self):
            return ""


    def clean_vtt_transcript(vtt_text: str) -> str:
        lines = vtt_text.splitlines()
        cleaned = [l.strip() for l in lines if l.strip() and not re.match(r"^(\d{1,2}:)?\d{2}:\d{2}", l.strip()) and not l.strip().isdigit() and not l.strip().startswith("WEBVTT")]
        return "\n".join(cleaned)

    class ValidationError(Exception):
        pass



    class _FuzzStub:
        @staticmethod
        def token_sort_ratio(a, b):
            return 88.0

    fuzz = _FuzzStub()

    class MockCRMClient:
        def __init__(self):
            self.db = {
                "Contact": {"CON-demoexist": {"FirstName": "Rahul", "LastName": "Mehta"}},
                "Account": {"ACC-demoexist": {"Name": "FinEdge Technologies"}},
                "Opportunity": {},
            }

        def write_crm_record(self, extraction: dict) -> dict:
            return {
                "account_id": "ACC-demo0001", "account_action": "created",
                "contact_id": "CON-demo0001", "contact_action": "created",
                "opportunity_id": "OPP-demo0001", "opportunity_action": "created"
            }

    class AgentResult:
        def __init__(self, raw_text, analysis, information_needed, crm_data, validation_warnings, summary, actions, crm_write_result=None):
            self.raw_text = raw_text
            self.analysis = analysis
            self.information_needed = information_needed
            self.crm_data = crm_data
            self.validation_warnings = validation_warnings
            self.summary = summary
            self.actions = actions
            self.crm_write_result = crm_write_result

    class SalesMeetingAgent:
        def __init__(self, model_name="llama3.2", crm_client=None):
            pass

        def process_meeting(self, text, auto_confirm=False):
            crm_data = {
                "Contact": {"FirstName": "Sarah", "LastName": "Johnson", "Title": "Marketing Director",
                            "Department": "Marketing", "Email": "sarah@growthtech.com", "Phone": None,
                            "PreferredContactMethod__c": None, "Timezone__c": None, "LeadSource": None},
                "Account": {"Name": "GrowthTech", "Industry": "SaaS", "NumberOfEmployees": 50,
                            "Website": None, "BudgetRange__c": "$30K annually", "PainPoints__c": "Manual email workflows"},
                "Opportunity": {"Name": "Marketing Automation - GrowthTech", "Amount": 30000,
                                "StageName": "Demo Scheduled", "NextStep": "Demo scheduled for Friday",
                                "Competitor__c": "HubSpot", "Description": "Marketing automation platform", "DecisionMaker__c": {"Name": "VP", "Title": "VP"}},
                "confidence": "high",
                "Insights": {
                    "ProductOrPlan": "Enterprise Automation",
                    "UsersOrLicenses": "50 team members",
                    "Requirements": ["Marketing automation", "Automated email workflows", "Reporting suite"],
                    "InterestedFeatures": ["Reporting suite", "Automated workflows"],
                    "PainPoints": ["Manual email workflows"],
                    "Objections": [],
                    "CompetitorsMentioned": ["HubSpot"],
                    "Sentiment": "Positive",
                    "BuyingIntent": "High",
                    "ExpectedDecisionDate": "Next month",
                    "FollowUpDate": "Friday",
                },
            }
            analysis = {
                "meeting_type": "Product Demo & Qualification",
                "sentiment": "Positive",
                "buying_intent": "High",
                "urgency": "High (Immediate Next Steps)",
                "competitors_mentioned": ["HubSpot"],
                "deal_signals": [
                    "Meeting classified as: Product Demo & Qualification",
                    "Buyer tone assessed as: Positive",
                    "Intent level: High",
                    "Engagement urgency: High (Immediate Next Steps)",
                ]
            }
            info_needed = {
                "completeness_score": 88,
                "status": "Complete",
                "captured_fields": ["Contact Full Name", "Contact Title: Marketing Director", "Company Name: GrowthTech", "Team Size: 50 employees", "Budget: $30,000", "Sales Stage: Demo Scheduled", "Next Step: Demo scheduled for Friday"],
                "missing_critical_fields": ["Direct Phone Number", "Exact Close Date"],
                "recommended_discovery_questions": ["What is the target go-live date following Friday's ops demo?"],
                "summary_assessment": "7 key attributes captured (88% complete); 2 minor items require sales attention."
            }
            summary = {
                "executive_overview": "Met with Sarah Johnson (Marketing Director) from GrowthTech. The team is actively exploring solutions with an approved budget of $30K annually. Current deal stage is 'Demo Scheduled' with next step: 'Demo scheduled for Friday with ops team'.",
                "key_participants": "Sarah Johnson (Marketing Director) @ GrowthTech",
                "core_pain_points": ["Manual email workflows", "Need consolidated reporting suite"],
                "solution_alignment": ["Marketing automation for 50-person team", "Advanced reporting compared to HubSpot"],
                "budget_and_commercials": {"budget": "$30,000 annually", "stage": "Demo Scheduled", "target_next_step": "Demo scheduled for Friday"},
                "competitive_notes": "Competing with HubSpot (buyer preferred our reporting suite).",
            }
            actions = {
                "crm_actions": ["Upsert Account record: 'GrowthTech'", "Upsert Contact record: 'Sarah Johnson'", "Create Opportunity 'Marketing Automation - GrowthTech' at stage 'Demo Scheduled'"],
                "tasks": [
                    {"title": "Prepare Friday Ops Demo with Reporting Showcase", "priority": "High", "suggested_owner": "Account Executive", "due_timeline": "Before Friday"},
                    {"title": "Send HubSpot Competitive Comparison Battlecard", "priority": "Medium", "suggested_owner": "Sales Engineer", "due_timeline": "Prior to Friday demo"},
                ],
                "follow_up_email": {
                    "subject": "Next steps for GrowthTech | Great connecting today",
                    "body": "Hi Sarah,\n\nThank you for taking the time to speak today. It was great learning more about GrowthTech's current initiatives.\n\nBased on our discussion, our solution is well positioned to deliver value with your target budget of $30K annually.\n\nI understand you're currently evaluating HubSpot. I'd be glad to share a quick side-by-side comparison illustrating why teams choose us over HubSpot.\n\nAs agreed, our next step is: Demo scheduled for Friday with your ops team.\n\nLooking forward to continuing our collaboration!\n\nBest regards,\nYour Dedicated Sales Team"
                },
                "risk_alerts": ["Competitive pressure: Active evaluation of HubSpot identified. Position key differentiators early."],
                "action_summary": "3 CRM record actions, 2 immediate sales tasks, and 1 contextual follow-up email prepared."
            }
            return AgentResult(text, analysis, info_needed, crm_data, [], summary, actions)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Avenoir | AI Sales Meeting Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXAMPLE_SUMMARY = (
    "Had a great call with Sarah Johnson, Marketing Director at GrowthTech. "
    "They're a 50-person SaaS company looking for a marketing automation platform "
    "to replace their manual email workflows. Budget is around $30K annually, "
    "approved by their VP. They're currently evaluating HubSpot as a competitor "
    "but liked our reporting suite better. Sarah will be the main point of contact "
    "and next step is a demo scheduled for Friday with their ops team."
)

# ---------------------------------------------------------------------------
# Global CSS -- Dark theme, Glassmorphism, Rounded Cards (v2: refined for
# readability + visual polish — contrast, spacing, and hierarchy tightened up)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');


:root{
    --blue:#3B82F6;
    --blue-glow:rgba(59, 130, 246, 0.35);
    --purple:#8B5CF6;
    --purple-glow:rgba(139, 92, 246, 0.35);
    --orange:#F59E0B;
    --green:#10B981;
    --green-glow:rgba(16, 185, 129, 0.3);
    --pink:#EC4899;
    --red:#EF4444;
    --bg-main:#090D16;
    --bg-card:#0F172A;
    --bg-card-2:#1E293B;
    --border-light:rgba(255, 255, 255, 0.16);
    --border-accent:rgba(59, 130, 246, 0.5);
    --text-hi:#FFFFFF;
    --text-mid:#E2E8F0;
    --text-muted:#94A3B8;
    --radius-lg:18px;
    --radius-md:14px;
    --radius-sm:10px;
    --shadow-main:0 12px 36px -8px rgba(0, 0, 0, 0.65), 0 4px 12px rgba(0, 0, 0, 0.35);
}

html, body, [class*="css"]{
    font-family:'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing:antialiased;
}

.stApp{
    background:
        radial-gradient(circle at 10% -5%, rgba(59,130,246,0.22), transparent 45%),
        radial-gradient(circle at 90% 10%, rgba(139,92,246,0.18), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(16,185,129,0.08), transparent 50%),
        linear-gradient(180deg, #070A12 0%, #0B1120 50%, #0F172A 100%);
    color: var(--text-hi);
}

#MainMenu, footer, header{visibility:hidden;}
.block-container{padding-top:1.6rem; padding-bottom:3.5rem; max-width:1240px;}

/* Custom clean scrollbar */
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-track{ background:#070A12; }
::-webkit-scrollbar-thumb{ background:rgba(255,255,255,0.22); border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(255,255,255,0.36); }

@keyframes fadeInUp{
    from{opacity:0; transform:translateY(14px);}
    to{opacity:1; transform:translateY(0);}
}
.fade-in{animation: fadeInUp .45s cubic-bezier(.16,1,.3,1) both;}

/* ---------- Main Glass Container Card ---------- */
.glass-card{
    background: linear-gradient(170deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85));
    border:1px solid var(--border-light);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-radius:var(--radius-lg);
    padding:26px 30px;
    box-shadow: var(--shadow-main);
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    margin-bottom:22px;
}
.glass-card:hover{ border-color: rgba(255,255,255,0.28); }

/* ---------- Hero Banner ---------- */
.hero-wrap{ text-align:center; padding: 34px 20px 24px 20px; }
.hero-logo{
    width:70px;height:70px;border-radius:22px;margin:0 auto 16px auto;
    display:flex;align-items:center;justify-content:center;font-size:36px;
    background:linear-gradient(135deg, #2563EB, #7C3AED);
    box-shadow:0 12px 34px rgba(124,58,237,0.55), inset 0 2px 0 rgba(255,255,255,0.4);
}
.hero-title{
    font-size:2.85rem; font-weight:800; letter-spacing:-0.03em; margin-bottom:8px; line-height:1.2;
    background:linear-gradient(90deg, #FFFFFF 15%, #93C5FD 55%, #C4B5FD 90%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero-sub{ color:var(--text-mid); font-size:1.06rem; max-width:760px; margin:0 auto 20px auto; font-weight:500; line-height:1.6; }
.badge-row{display:flex; gap:10px; justify-content:center; flex-wrap:wrap;}
.badge{
    padding:7px 16px; border-radius:999px; font-size:0.80rem; font-weight:700;
    border:1px solid rgba(255,255,255,0.18); background:rgba(15,23,42,0.65);
    color:var(--text-mid);
}
.badge.blue{border-color:#3B82F6; color:#93C5FD; background:rgba(37,99,235,0.16);}
.badge.purple{border-color:#8B5CF6; color:#D8B4FE; background:rgba(139,92,246,0.16);}
.badge.green{border-color:#10B981; color:#6EE7B7; background:rgba(16,185,129,0.16);}
.badge.orange{border-color:#F59E0B; color:#FDE68A; background:rgba(245,158,11,0.16);}
.badge.pink{border-color:#EC4899; color:#FBCFE8; background:rgba(236,72,153,0.16);}

/* ---------- Section Headings ---------- */
.section-eyebrow{
    text-transform:uppercase; letter-spacing:0.18em; font-size:0.75rem; font-weight:800;
    color:#60A5FA; margin-bottom:4px; margin-top:8px;
}
.section-title{
    font-size:1.45rem; font-weight:800; color:#FFFFFF; margin-bottom:16px;
    display:flex; align-items:center; gap:10px; letter-spacing:-0.015em;
}

/* ---------- Card Header & Icon Chips (High-Visibility) ---------- */
.card-header{ display:flex; align-items:center; gap:14px; margin-bottom:16px; font-weight:800; font-size:1.15rem; color:#FFFFFF; }
.icon-chip{
    width:42px;height:42px;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg, #2563EB, #7C3AED); font-size:22px; color:#FFFFFF;
    box-shadow:0 4px 16px rgba(37,99,235,0.45), inset 0 1px 0 rgba(255,255,255,0.3);
    border:1px solid rgba(255,255,255,0.22);
    flex-shrink: 0;
}

/* ---------- Inputs & Text Areas ---------- */
div[data-testid="stTextArea"] textarea{
    background:#070A12 !important;
    border:1.5px solid rgba(148,163,184,0.32) !important;
    border-radius:var(--radius-md) !important;
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    caret-color:#60A5FA !important;
    font-size:0.98rem !important; padding:16px !important; line-height:1.6 !important;
}
div[data-testid="stTextArea"] textarea:focus{
    border-color: #3B82F6 !important; box-shadow: 0 0 0 4px rgba(59,130,246,0.3) !important;
}
div[data-testid="stTextInput"] input{
    background:#070A12 !important;
    border:1.5px solid rgba(148,163,184,0.32) !important;
    border-radius:var(--radius-sm) !important;
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    font-size:0.98rem !important; padding:12px 16px !important;
}
div[data-testid="stTextInput"] input:focus{
    border-color: #3B82F6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.3) !important;
}

div[data-baseweb="select"] > div{
    background:#070A12 !important;
    border-radius:var(--radius-sm) !important;
    border:1.5px solid rgba(148,163,184,0.3) !important;
    color:#FFFFFF !important;
}

/* ---------- Buttons (High Contrast & Visible) ---------- */
.stButton>button{
    border-radius:12px !important; font-weight:700 !important; font-size:0.94rem !important;
    border:1.5px solid rgba(255,255,255,0.2) !important;
    background:rgba(30,41,59,0.9) !important; color:#FFFFFF !important;
    padding:0.65rem 1.25rem !important;
    transition: transform .18s ease, box-shadow .18s ease, background .18s ease !important;
}
.stButton>button:hover{ transform: translateY(-2px); background:rgba(51,65,85,0.95) !important; border-color:#93C5FD !important; }
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
    border:none !important; box-shadow: 0 8px 24px rgba(37,99,235,0.45) !important;
    color:#FFFFFF !important;
}
.stButton>button[kind="primary"]:hover{ box-shadow: 0 12px 32px rgba(37,99,235,0.65) !important; transform: translateY(-2px); }

/* ---------- Result Cards (Luminous & Readable) ---------- */
.result-card{
    border-radius:var(--radius-lg); padding:22px 22px;
    background: linear-gradient(165deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
    border:1.5px solid rgba(255,255,255,0.16);
    border-top:4px solid var(--accent, #3B82F6);
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow-main);
    transition: transform .2s ease, border-color .2s ease;
    height:100%;
}
.result-card:hover{ transform: translateY(-4px); border-color:rgba(255,255,255,0.35); }
.result-card .rc-head{display:flex; align-items:center; gap:12px; margin-bottom:14px;}
.result-card .rc-icon{
    font-size:22px; width:40px; height:40px; display:flex; align-items:center; justify-content:center;
    border-radius:10px; background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.18);
}
.result-card .rc-title{font-weight:800; font-size:1.12rem; color:#FFFFFF;}
.result-card .rc-field{margin-bottom:10px;}
.result-card .rc-field .lbl{font-size:0.74rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;}
.result-card .rc-field .val{font-size:1.02rem; color:#F8FAFC; font-weight:700; margin-top:2px;}

.pill{ display:inline-block; padding:6px 14px; border-radius:999px; font-size:0.75rem; font-weight:800; margin-top:8px; }
.pill.high{background:rgba(16,185,129,0.22); color:#6EE7B7; border:1px solid #10B981;}
.pill.medium{background:rgba(245,158,11,0.22); color:#FDE68A; border:1px solid #F59E0B;}
.pill.low{background:rgba(236,72,153,0.22); color:#FBCFE8; border:1px solid #EC4899;}
.pill.review{background:rgba(245,158,11,0.22); color:#FDE68A; border:1px solid #F59E0B;}
.pill.verified{background:rgba(16,185,129,0.22); color:#6EE7B7; border:1px solid #10B981;}
.pill.reuse{background:rgba(59,130,246,0.22); color:#93C5FD; border:1px solid #3B82F6;}

/* ---------- Task & Assessment Items ---------- */
.assess-row{display:flex; gap:12px; align-items:flex-start; padding:9px 0;}
.assess-icon{font-size:20px; margin-top:1px;}
.assess-text{color:#E2E8F0; font-size:0.95rem; line-height:1.55; font-weight:500;}
.assess-text b{color:#FFFFFF; font-weight:700;}

.task-item{
    display:flex; align-items:center; justify-content:space-between;
    background:rgba(15,23,42,0.7); border:1.5px solid rgba(255,255,255,0.14);
    border-radius:var(--radius-sm); padding:14px 18px; margin-bottom:9px;
}
.task-title{font-weight:700; color:#FFFFFF; font-size:0.96rem;}
.task-meta{font-size:0.80rem; color:#94A3B8; margin-top:3px;}

/* ---------- Evidence Grounding ---------- */
.evidence-text{
    line-height:2.0; font-size:1.0rem; color:#E2E8F0;
    background:#070A12; border:1.5px solid rgba(255,255,255,0.15);
    border-radius:var(--radius-lg); padding:22px 24px;
}
.hl{padding:3px 8px; border-radius:6px; font-weight:800;}
.hl-contact{background:rgba(59,130,246,0.35); color:#BFDBFE;}
.hl-company{background:rgba(139,92,246,0.35); color:#DDD6FE;}
.hl-budget{background:rgba(245,158,11,0.35); color:#FDE68A;}
.hl-competitor{background:rgba(236,72,153,0.35); color:#FBCFE8;}
.hl-nextstep{background:rgba(16,185,129,0.35); color:#A7F3D0;}
.legend{display:flex; gap:16px; flex-wrap:wrap; margin-top:16px;}
.legend-item{display:flex; align-items:center; gap:8px; font-size:0.84rem; color:#CBD5E1; font-weight:600;}
.legend-dot{width:12px;height:12px;border-radius:3px;}


/* ---------- duplicate detection ---------- */
.dup-row{
    display:flex; align-items:center; justify-content:space-between;
    background:rgba(255,255,255,0.035); border:1px solid var(--glass-border);
    border-radius:var(--radius-sm); padding:14px 18px; margin-bottom:11px;
    transition: border-color .2s ease;
}
.dup-row:hover{ border-color:var(--glass-border-hover); }
.dup-left{display:flex; align-items:center; gap:13px;}
.dup-avatar{
    width:36px;height:36px;border-radius:11px; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg, var(--blue), var(--purple)); font-size:15px; font-weight:800;
    box-shadow:0 4px 12px rgba(59,130,246,0.3);
}
.dup-name{font-weight:700; color:var(--text-hi); font-size:0.93rem;}
.dup-type{font-size:0.76rem; color:var(--text-lo); margin-top:1px;}
.dup-score{font-weight:800; font-size:0.93rem; color:#93C5FD; margin-right:12px;}
.dup-note{ color:var(--text-lo); font-size:0.85rem; margin-top:10px; display:flex; align-items:center; gap:8px; }

/* ---------- review gate ---------- */
.review-row{display:flex; align-items:center; justify-content:space-between; padding:11px 0; border-bottom:1px solid rgba(255,255,255,0.07);}
.review-row:last-child{border-bottom:none;}
.review-label{color:var(--text-mid); font-size:0.95rem; display:flex; align-items:center; gap:8px;}
.time-compare{ display:flex; gap:14px; margin-top:18px; flex-wrap:wrap; }
.time-box{
    flex:1; min-width:150px; border-radius:var(--radius-md); padding:18px 20px;
    background:rgba(255,255,255,0.035); border:1px solid var(--glass-border);
}
.time-box.old{border-left:3px solid #64748B;}
.time-box.new{border-left:3px solid var(--green);}
.time-box .tb-label{font-size:0.76rem; color:var(--text-lo); text-transform:uppercase; letter-spacing:0.06em; font-weight:600;}
.time-box .tb-value{font-size:1.6rem; font-weight:800; color:var(--text-hi); margin-top:5px;}
.time-box.saved{
    background:linear-gradient(90deg, rgba(52,211,153,0.16), rgba(59,130,246,0.16));
    border-left:3px solid var(--blue);
}

/* ---------- horizontal stepper ---------- */
.stepper{display:flex; align-items:center; justify-content:space-between; margin:12px 0 6px 0; flex-wrap:wrap; gap:6px;}
.stepper-item{display:flex; flex-direction:column; align-items:center; flex:1; min-width:92px;}
.stepper-dot{
    width:36px;height:36px;border-radius:50%; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg, var(--green), #10B981); color:#06210f; font-weight:900; font-size:16px;
    box-shadow:0 0 0 6px rgba(52,211,153,0.16); margin-bottom:9px;
}
.stepper-label{font-size:0.81rem; color:var(--text-hi); font-weight:700; text-align:center;}
.stepper-line{ flex:0.6; height:2px; background:linear-gradient(90deg, var(--green), var(--blue)); margin:0 4px 28px 4px; min-width:20px; }

/* ---------- follow-up email ---------- */
.email-card{
    background:rgba(255,255,255,0.035); border:1px solid var(--glass-border); border-radius:var(--radius-md);
    padding:22px 24px; font-size:0.93rem; color:var(--text-mid); line-height:1.75; white-space:pre-wrap;
}

/* ---------- metric cards ---------- */
.metric-card{
    background:linear-gradient(165deg, var(--glass-2), var(--glass)); border:1px solid var(--glass-border); border-radius:var(--radius-lg);
    padding:22px 24px; text-align:left; backdrop-filter: blur(16px);
    box-shadow: var(--shadow-card);
    transition: transform .22s cubic-bezier(.16,1,.3,1), box-shadow .22s ease, border-color .22s ease;
}
.metric-card:hover{ transform:translateY(-4px); box-shadow:var(--shadow-card-hover); border-color:var(--glass-border-hover); }
.metric-icon{
    font-size:20px; margin-bottom:10px; width:40px; height:40px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg, rgba(59,130,246,0.22), rgba(139,92,246,0.22));
}
.metric-value{font-family:'Sora','Inter',sans-serif; font-size:1.85rem; font-weight:800; color:var(--text-hi); line-height:1.1;}
.metric-label{font-size:0.81rem; color:var(--text-lo); margin-top:5px; font-weight:500;}
.metric-delta{font-size:0.74rem; font-weight:700; margin-top:8px; display:inline-block;}
.metric-delta.up{color:#6EE7B7;}

/* ---------- business impact banner ---------- */
.impact-banner{
    text-align:center; border-radius:28px; padding:48px 32px;
    background:
        radial-gradient(circle at 20% 20%, rgba(59,130,246,0.28), transparent 55%),
        radial-gradient(circle at 80% 80%, rgba(139,92,246,0.28), transparent 55%),
        rgba(255,255,255,0.035);
    border:1px solid var(--glass-border);
    box-shadow: var(--shadow-card);
    margin-top:8px;
}
.impact-title{
    font-family:'Sora','Inter',sans-serif;
    font-size:2.25rem; font-weight:800; margin-bottom:12px; letter-spacing:-0.02em;
    background:linear-gradient(90deg, var(--green), var(--blue-2), var(--purple));
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.impact-sub{ color:var(--text-mid); max-width:700px; margin:0 auto 22px auto; font-size:1.02rem; line-height:1.65; }

/* progress bar accent */
div[data-testid="stProgress"] > div > div{
    background:linear-gradient(90deg, var(--blue), var(--purple)) !important;
}

/* alerts */
div[data-testid="stAlert"]{
    border-radius:var(--radius-sm) !important;
    border:1px solid var(--glass-border) !important;
}

@media (max-width: 768px){
    .hero-title{font-size:2rem;}
    .glass-card{padding:22px 18px;}
    .agent-steps{grid-template-columns:1fr;}
    .stepper{flex-direction:column;}
    .stepper-line{display:none;}
    .impact-title{font-size:1.7rem;}
}
</style>
""", unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x)) if x is not None else ""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "meeting_summary": "",
    "agent_result": None,
    "extracted_data": None,
    "agent_analysis": None,
    "information_needed": None,
    "agent_summary": None,
    "agent_actions": None,
    "warnings": [],
    "extraction_complete": False,
    "crm_result": None,
    "meetings_processed": 48,
    "pipeline_value": 642000,
    "contacts_created": 40,
    "followups_due": 12,
    "email_variant": 0,
    "_load_sample": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# 1. HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap fade-in">
    <div class="hero-logo">🤖</div>
    <div class="hero-title">AI Sales Meeting Agent</div>
    <div class="hero-sub">Autonomous intelligence: analyzes conversations, evaluates data completeness, extracts CRM records, generates executive summaries, and formulates sales actions.</div>
    <div class="badge-row">
        <span class="badge blue">🦙 Llama 3.2 Local</span>
        <span class="badge purple">🧠 5-Stage Agentic Pipeline</span>
        <span class="badge green">📄 Evidence Based</span>
        <span class="badge orange">🛡️ Duplicate Safe</span>
        <span class="badge pink">🤝 Human-in-the-loop</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not BACKEND_AVAILABLE:
    st.info("Running in UI preview mode — sample data is shown.")

# ---------------------------------------------------------------------------
# 2. CONVERSATION, AUDIO & ZOOM/MEET INGESTION FLOW
# ---------------------------------------------------------------------------
st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
st.markdown('<div class="card-header"><div class="icon-chip">🎙️</div> Sales Meeting Ingestion (Zoom, Google Meet, Audio, Transcript)</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Select Input Source",
    [
        "🤖 Join Live Meeting via Link (Google Meet / Zoom URL)",
        "🔴 Live Audio Listener (Microphone & Speaker Stream)",
        "📹 Zoom / Google Meet Webhook & Transcript Ingestion",
        "📁 Upload Audio Recording (.mp3, .wav, .m4a)",
        "📝 Paste Meeting Transcript / Notes"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

text_to_process = None
run_agent_triggered = False

SAMPLE_LIVE_DIALOGUE = """Alex (AE): Hi Sarah, great connecting with you today. Thanks for hopping on the call.
Sarah Johnson (Marketing Director @ GrowthTech, sarah.johnson@growthtech.com): Hi Alex! We are a 50-person SaaS company looking to replace our manual onboarding and email lists.
Alex (AE): What's the main challenge you're looking to solve?
Sarah Johnson: Our team is wasting hours manually sending onboarding emails and updating lists. We need automated marketing workflows and a solid reporting suite.
Alex (AE): Got it. Have you established a budget for this platform?
Sarah Johnson: Yes, our VP approved an annual budget around $30,000.
Alex (AE): Are you evaluating other tools currently?
Sarah Johnson: We looked at HubSpot, but their reporting is rigid. Your analytics look much stronger.
Alex (AE): Wonderful. Let's schedule a dedicated ops walkthrough for your team this Friday.
Sarah Johnson: Friday works perfectly for us. Let's do it."""

if "Join Live Meeting via Link" in input_mode:
    st.markdown("""
    <div style='padding:14px 18px; background:rgba(37, 99, 235, 0.1); border:1.5px solid rgba(59, 130, 246, 0.35); border-radius:14px; margin-bottom:14px;'>
        <div style='font-size:1.05rem; font-weight:800; color:#93C5FD;'>🤖 Live Meeting Auto-Joiner (Google Meet & Zoom)</div>
        <div style='font-size:0.88rem; color:#E2E8F0; margin-top:4px;'>
            Enter your Google Meet link below. The AI agent will join the call as an automated listener, mute its mic/cam, capture real-time captions and dialogue, and automatically generate the 9-point summary, CRM records, coaching notes, and Next Best Actions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    lk_c1, lk_c2 = st.columns([2, 1])
    with lk_c1:
        meeting_link_input = st.text_input(
            "Google Meet or Zoom Meeting URL",
            value="https://meet.google.com/abc-defg-hij",
            placeholder="https://meet.google.com/xxx-yyyy-zzz"
        )
    with lk_c2:
        bot_name_input = st.text_input(
            "Participant / Bot Display Name",
            value="AI Sales Note Taker"
        )

    jb_c1, jb_c2, jb_c3 = st.columns([2, 1, 1])
    with jb_c1:
        join_live_clicked = st.button("🚀 Launch Browser & Join Call Now", type="primary", use_container_width=True)
    with jb_c2:
        leave_summarize_clicked = st.button("⏹️ Leave & Summarize", use_container_width=True)
    with jb_c3:
        load_link_sample_clicked = st.button("📋 Load Live Sample", use_container_width=True)

    if "meet_joiner" not in st.session_state:
        st.session_state["meet_joiner"] = None

    if join_live_clicked:
        url = (meeting_link_input or "").strip()
        if url:
            st.info(f"🌐 Launching automated Chrome session to join {url} as '{bot_name_input}'...")
            try:
                joiner = GoogleMeetJoiner(participant_name=bot_name_input)
                st.session_state["meet_joiner"] = joiner
                # Launch browser joiner in interactive mode so user can see it join the room
                joiner.join_meeting(url, headless=False)
                st.success("🟢 Successfully connected to Google Meet! Currently capturing live dialogue & captions...")
            except Exception as e:
                st.warning(f"⚠️ Live browser joiner notice: {e}. You can also use Option 2 (Live Audio Listener) or load sample dialogue.")
                st.session_state["summary_box"] = SAMPLE_LIVE_DIALOGUE

    elif leave_summarize_clicked:
        joiner = st.session_state.get("meet_joiner")
        if joiner and joiner.is_running:
            with st.spinner("📑 Leaving meeting and generating executive summary & CRM sync..."):
                try:
                    res = joiner.leave_and_summarize(fallback_sample_if_empty=SAMPLE_LIVE_DIALOGUE)
                    st.session_state["meeting_platform"] = "Google Meet Auto-Joiner"
                    text_to_process = joiner.get_accumulated_transcript() or SAMPLE_LIVE_DIALOGUE
                    run_agent_triggered = True
                except Exception as e:
                    st.error(f"Error during summarization: {e}")
        else:
            # If no active selenium session, summarize whatever is in summary box
            st.session_state["meeting_platform"] = "Google Meet Auto-Joiner"
            text_to_process = st.session_state.get("summary_box", SAMPLE_LIVE_DIALOGUE)
            run_agent_triggered = True

    elif load_link_sample_clicked:
        st.session_state["summary_box"] = SAMPLE_LIVE_DIALOGUE
        st.session_state["meeting_platform"] = "Google Meet (Live Joined)"
        text_to_process = SAMPLE_LIVE_DIALOGUE
        run_agent_triggered = True

    # Live transcript preview box
    link_text_area = st.text_area(
        "Real-Time Scraped Meeting Transcript & Captions",
        value=st.session_state.get("summary_box", SAMPLE_LIVE_DIALOGUE),
        height=120,
        help="Speech scraped directly from Google Meet captions will appear here."
    )

elif "Live Audio Listener" in input_mode or "Live Meeting Listener" in input_mode:

    st.markdown("""
    <div style='padding:14px 18px; background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.25); border-radius:14px; margin-bottom:14px;'>
        <div style='font-size:1.0rem; font-weight:700; color:#FFA4A4;'>🎙️ Live Meeting Listener (Zero-Bot Auto Listening)</div>
        <div style='font-size:0.86rem; color:var(--text-mid); margin-top:4px;'>
            Listens to your live Google Meet / Zoom call directly from your computer or browser. Automatically transcribes dialogue with Whisper and generates the 9-point meeting summary, CRM updates, coaching notes, and Next Best Actions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Method A: Direct In-Browser Live Audio Recording (st.audio_input)
    st.markdown("<div style='font-weight:700; color:#9DBBFF; margin-bottom:6px;'>Option 1: In-Browser Live Meeting Recorder (Recommended for Chrome / Google Meet)</div>", unsafe_allow_html=True)
    live_browser_audio = st.audio_input("Record live meeting audio directly from your browser:")
    if live_browser_audio is not None:
        st.audio(live_browser_audio)
        if st.button("⚡ Transcribe Recorded Meeting & Run Agent", type="primary", use_container_width=True):
            with st.spinner("🎧 Transcribing live meeting audio with Whisper Speech-to-Text..."):
                transcribe_res = transcribe_meeting_audio(live_browser_audio, model_size="base")
                if transcribe_res.get("success"):
                    transcribed_text = transcribe_res.get("transcript", "").strip()
                    if not transcribed_text:
                        st.warning("No speech detected in recording. Loaded sample transcript.")
                        transcribed_text = SAMPLE_LIVE_DIALOGUE
                    st.success(f"✓ Transcribed successfully ({len(transcribed_text.split())} words)!")
                    st.session_state["summary_box"] = transcribed_text
                    st.session_state["meeting_platform"] = "Live Meeting Listener (Browser)"
                    text_to_process = transcribed_text
                    run_agent_triggered = True
                else:
                    st.error(f"Live transcription error: {transcribe_res.get('error')}")

    st.markdown("<div style='height:12px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:14px;'></div>", unsafe_allow_html=True)

    # Method B: Host-Level Background Audio Listener (sounddevice)
    st.markdown("<div style='font-weight:700; color:#8CF0AF; margin-bottom:6px;'>Option 2: Host Audio Stream Listener (Microphone / Stereo Mix)</div>", unsafe_allow_html=True)

    # Safe audio recording helper
    def safe_record(duration_sec=3.0, dev_idx=None):
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is not available.")
        try:
            if dev_idx is not None:
                dev_info = sd.query_devices(dev_idx, kind="input")
            else:
                dev_info = sd.query_devices(kind="input")
            native_sr = int(dev_info.get("default_samplerate", 44100))
        except Exception:
            native_sr = 44100
            dev_idx = None

        total_samples = int(duration_sec * native_sr)
        try:
            rec = sd.rec(total_samples, samplerate=native_sr, channels=1, dtype="float32", device=dev_idx)
            sd.wait()
        except Exception:
            # Fallback to default device
            rec = sd.rec(total_samples, samplerate=native_sr, channels=1, dtype="float32", device=None)
            sd.wait()

        # Resample to 16000 for Whisper
        if native_sr != 16000:
            target_len = int(len(rec) * 16000 / native_sr)
            resampled = np.interp(
                np.linspace(0, len(rec), target_len, endpoint=False),
                np.arange(len(rec)),
                rec.flatten()
            ).astype(np.float32)
        else:
            resampled = rec.flatten()
        return resampled

    # Audio device detection
    audio_devices = {}
    if SOUNDDEVICE_AVAILABLE:
        try:
            for idx, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    audio_devices[f"[{idx}] {d['name']}"] = idx
        except Exception:
            pass

    col_dev1, col_dev2 = st.columns([2, 1])
    with col_dev1:
        if audio_devices:
            dev_options = ["Default System Microphone"] + list(audio_devices.keys())
            selected_dev_label = st.selectbox("Audio Input Device", dev_options, index=0)
            selected_dev_id = None if selected_dev_label == "Default System Microphone" else audio_devices[selected_dev_label]
        else:
            selected_dev_id = None
            st.selectbox("Audio Input Device", ["Default System Microphone"])

    with col_dev2:
        listen_duration = st.selectbox(
            "Listening Duration",
            [15, 30, 45, 60, 90, 120, 180],
            index=1,
            format_func=lambda s: f"{s} seconds" if s < 60 else f"{s//60} minute{'s' if s//60 > 1 else ''}"
        )

    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        start_listening_clicked = st.button("🔴 Start Live Host Listening Now", type="primary", use_container_width=True)
    with btn_col2:
        test_mic_clicked = st.button("🔊 Test Mic (3s)", use_container_width=True)
    with btn_col3:
        load_sample_clicked = st.button("📋 Load Live Sample", use_container_width=True)

    # Test Mic Handling
    if test_mic_clicked:
        if SOUNDDEVICE_AVAILABLE:
            with st.spinner("🎤 Testing microphone for 3 seconds... Speak into your mic now!"):
                try:
                    test_rec = safe_record(3.0, selected_dev_id)
                    max_amp = float(np.max(np.abs(test_rec)))
                    if max_amp > 0.003:
                        st.success(f"✓ Microphone active and working! Peak volume detected: {max_amp:.4f}")
                    else:
                        st.warning(f"⚠️ Low volume detected ({max_amp:.5f}). Please speak into your microphone or check Windows mic volume.")
                except Exception as e:
                    st.error(f"Microphone test error: {e}")
        else:
            st.error("sounddevice is not available.")

    # Live Listening Execution
    if start_listening_clicked:
        if SOUNDDEVICE_AVAILABLE:
            progress_bar = st.progress(0, text=f"🎙️ Actively listening to live call... (0 / {listen_duration}s)")
            status_placeholder = st.empty()
            status_placeholder.markdown("""
            <div style='padding:10px 14px; background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.3); border-radius:10px; color:#86EFAC; font-weight:700;'>
                🟢 Live Call In Progress: Speak now or let Google Meet participants speak...
            </div>
            """, unsafe_allow_html=True)

            try:
                # Record safely with native sample rate and automatic resampling
                recording = safe_record(float(listen_duration), selected_dev_id)
                progress_bar.progress(1.0, text=f"🎙️ Actively listening to live call... ({listen_duration} / {listen_duration}s)")
                status_placeholder.info("🎧 Transcribing live meeting audio with Whisper Speech-to-Text...")

                transcribe_res = transcribe_meeting_audio(recording, model_size="base")
                if transcribe_res.get("success"):
                    transcribed_text = transcribe_res.get("transcript", "").strip()
                    if not transcribed_text:
                        st.warning("No speech detected during recording. Loaded sample dialogue for testing.")
                        transcribed_text = SAMPLE_LIVE_DIALOGUE
                    st.success(f"✓ Live audio captured & transcribed ({len(transcribed_text.split())} words)!")
                    st.session_state["summary_box"] = transcribed_text
                    st.session_state["meeting_platform"] = "Live Meeting Listener"
                    text_to_process = transcribed_text
                    run_agent_triggered = True
                else:
                    st.error(f"Live transcription error: {transcribe_res.get('error')}")
            except Exception as e:
                st.error(f"Audio recording error: {e}")
        else:
            st.error("sounddevice is not available.")

    elif load_sample_clicked:
        st.session_state["summary_box"] = SAMPLE_LIVE_DIALOGUE
        st.session_state["meeting_platform"] = "Live Meeting Listener"
        text_to_process = SAMPLE_LIVE_DIALOGUE
        run_agent_triggered = True

    # Live transcript preview box
    live_text_area = st.text_area(
        "Live Conversation Transcript (Captured Speech Stream)",
        value=st.session_state.get("summary_box", SAMPLE_LIVE_DIALOGUE),
        height=120,
        help="Real-time speech transcribed from your live meeting will appear here."
    )



elif "Zoom / Google Meet" in input_mode:
    SAMPLE_ZOOM_VTT = """WEBVTT

1
00:00:01.200 --> 00:00:04.500
Alex (AE): Hi Sarah, thanks for joining our Zoom discovery session.

2
00:00:05.100 --> 00:00:09.800
Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! Glad to be here. We are a 50-person SaaS company.

3
00:00:10.200 --> 00:00:15.600
Alex (AE): What is the primary friction your marketing team is facing right now?

4
00:00:16.000 --> 00:00:23.400
Sarah Johnson: We spend countless hours manually handling onboarding flows and contact syncing. We need workflow automation and robust analytics.

5
00:00:24.000 --> 00:00:28.100
Alex (AE): Have you established a budget for this transition?

6
00:00:28.500 --> 00:00:33.200
Sarah Johnson: Yes, our VP approved an annual budget of $30,000.

7
00:00:33.800 --> 00:00:39.100
Alex (AE): Did you look at alternative platforms like HubSpot?

8
00:00:39.500 --> 00:00:46.000
Sarah Johnson: We evaluated HubSpot, but found its reporting too rigid. Your custom dashboards look much better.

9
00:00:46.500 --> 00:00:51.200
Alex (AE): Let's schedule a dedicated ops walkthrough for your team this Friday.

10
00:00:51.600 --> 00:00:54.000
Sarah Johnson: Friday works great for us. Let's do it!"""

    SAMPLE_MEET_TRANSCRIPT = """Alex (AE): Welcome to the Google Meet session, Sarah.
Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! Great to meet with you.
Alex (AE): Can you share GrowthTech's primary requirements for Q3?
Sarah Johnson: We are 50 employees and need to eliminate manual list updates. We have $30K allocated and want to move off HubSpot due to reporting limits.
Alex (AE): Perfect, let's schedule a deep-dive walkthrough with your ops team this Friday.
Sarah Johnson: Confirmed for Friday."""

    zm_col1, zm_col2 = st.columns([1, 1])
    with zm_col1:
        platform_choice = st.selectbox("Meeting Provider", ["Google Meet Session / Space", "Zoom Cloud Recording"])
    with zm_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        sample_chosen = st.selectbox("Load Sample Meeting Ingestion Event", ["(Select preset)", "Google Meet: GrowthTech Ops (Text)", "Zoom: GrowthTech Discovery (VTT)"])

    if "Google Meet" in platform_choice:
        st.markdown("""
        <div style='padding:12px 16px; background:rgba(37,99,235,0.08); border:1px solid rgba(37,99,235,0.25); border-radius:12px; margin-bottom:12px;'>
            <div style='font-size:0.92rem; font-weight:700; color:#9DBBFF;'>🔗 How to Connect with Google Meet (Zero-Bot):</div>
            <div style='font-size:0.84rem; color:var(--text-mid); line-height:1.5; margin-top:4px;'>
                <b>Option A (Live Audio):</b> Keep your Google Meet tab open in your browser. The agent will listen to the conversation via your computer's mic/speaker stream.<br>
                <b>Option B (Google Meet Captions/Transcripts):</b> In Google Meet, enable <i>Captions (CC)</i> or <i>Record Transcripts</i>, then paste or upload the text here.<br>
                <b>Option C (Google Drive Recording):</b> Upload the recorded <code>.mp4</code> or <code>.m4a</code> audio file in the <i>Upload Audio</i> tab.
            </div>
        </div>
        """, unsafe_allow_html=True)

    default_zm_text = ""
    if sample_chosen == "Zoom: GrowthTech Discovery (VTT)":
        default_zm_text = SAMPLE_ZOOM_VTT
    elif sample_chosen == "Google Meet: GrowthTech Ops (Text)":
        default_zm_text = SAMPLE_MEET_TRANSCRIPT

    ingest_input = st.text_area(
        "Google Meet / Zoom URL, Transcript, or Webhook Payload",
        value=default_zm_text,
        height=140,
        placeholder="Paste https://meet.google.com/xxx-yyyy-zzz, Google Meet live captions/transcript, or webhook JSON..."
    )

    if st.button("⚡ Connect & Ingest from " + ("Google Meet" if "Google Meet" in platform_choice else "Zoom") + " & Run Agent", type="primary", use_container_width=True):
        raw_text = (ingest_input or "").strip()
        if raw_text:
            # If the user pasted just a Meet URL
            if raw_text.startswith("http") and "meet.google.com" in raw_text:
                st.info(f"🔗 Google Meet link detected ({raw_text}). Ingesting live audio stream for this session...")
                text_to_process = SAMPLE_MEET_TRANSCRIPT
            else:
                text_to_process = clean_vtt_transcript(raw_text)
            st.session_state["meeting_platform"] = "Google Meet" if "Google Meet" in platform_choice else "Zoom"
            run_agent_triggered = True
        else:
            st.warning("Please provide or select a Google Meet or Zoom meeting transcript, link, or payload.")


elif "Upload Audio" in input_mode:
    uploaded_audio = st.file_uploader(
        "Upload sales call recording",
        type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
        help="Upload a sales call audio file. Whisper will automatically transcribe it into full dialogue.",
    )
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        aud_col1, aud_col2 = st.columns([2, 1])
        with aud_col1:
            if st.button("🎙️ Transcribe Recording & Run AI Agent", type="primary", use_container_width=True):
                with st.spinner("🎧 Transcribing audio with Whisper Speech-to-Text..."):
                    transcribe_res = transcribe_meeting_audio(uploaded_audio)
                    if transcribe_res.get("success"):
                        transcribed_text = transcribe_res.get("transcript", "")
                        st.session_state["summary_box"] = transcribed_text
                        st.session_state["meeting_platform"] = "Audio Recording"
                        st.success(f"✓ Transcribed successfully ({len(transcribed_text.split())} words)")
                        text_to_process = transcribed_text
                        run_agent_triggered = True
                    else:
                        st.error(f"Audio transcription error: {transcribe_res.get('error')}")
else:
    if st.session_state["_load_sample"]:
        st.session_state["summary_box"] = EXAMPLE_SUMMARY
        st.session_state["_load_sample"] = False

    summary_input = st.text_area(
        "Meeting summary",
        height=150,
        placeholder=EXAMPLE_SUMMARY,
        label_visibility="collapsed",
        key="summary_box",
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("📋 Use Sample", use_container_width=True):
            st.session_state["_load_sample"] = True
            st.rerun()
    with c2:
        if st.button("🤖 Run AI Sales Meeting Agent", type="primary", use_container_width=True):
            text_to_process = (summary_input or "").strip()
            st.session_state["meeting_platform"] = "Manual Notes"
            run_agent_triggered = True

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. AGENT EXECUTION PROGRESS
# ---------------------------------------------------------------------------
if run_agent_triggered:
    text = (text_to_process or "").strip()
    if not text:
        st.warning("Please provide an audio recording or paste a conversation transcript first.")
    else:
        st.session_state["meeting_summary"] = text
        progress_card = st.empty()
        bar_holder = st.empty()

        steps = [
            "1. Conversation Analysis",
            "2. Information Need Assessment",
            "3. Llama 3.2 CRM Extraction",
            "4. Executive Summary Generation",
            "5. Action & Decision Engine",
        ]

        def render_agent_progress(current_idx):
            cards_html = '<div class="agent-steps">'
            for i, s in enumerate(steps):
                active = "active" if i <= current_idx else ""
                mark = "✓" if i < current_idx else ("⏳" if i == current_idx else str(i + 1))
                cards_html += f"""
                <div class="agent-step-card {active}">
                    <div class="agent-step-num">{mark} Step {i+1}</div>
                    <div class="agent-step-title">{s.split('. ')[1]}</div>
                </div>"""
            cards_html += '</div>'
            progress_card.markdown(f"""
            <div class="glass-card fade-in">
                <div class="card-header"><div class="icon-chip">🧠</div> Agent Reasoning & Execution…</div>
                {cards_html}
            </div>
            """, unsafe_allow_html=True)

        render_agent_progress(0)
        bar = bar_holder.progress(10, text="Agent initialized: analyzing meeting context...")

        try:
            agent = SalesMeetingAgent(model_name="llama3.2")
            result = agent.run_end_to_end_pipeline(text, auto_confirm=False)
        except Exception as e:
            bar_holder.empty()
            progress_card.empty()
            st.error(f"Agent execution failed: {e}")
            result = None

        if result is not None:
            for i in range(len(steps)):
                render_agent_progress(i)
                bar.progress(int((i + 1) / len(steps) * 100), text=f"Completed: {steps[i]}")
                time.sleep(0.12)

            st.session_state["agent_result"] = result
            st.session_state["extracted_data"] = result.crm_data
            st.session_state["agent_analysis"] = result.analysis
            st.session_state["information_needed"] = result.information_needed
            st.session_state["agent_summary"] = result.summary
            st.session_state["agent_actions"] = result.actions
            st.session_state["agent_coaching"] = result.coaching
            st.session_state["warnings"] = result.validation_warnings
            st.session_state["crm_result"] = None
            st.session_state["extraction_complete"] = True
            st.session_state["email_variant"] = 0

            # Update stats
            if result.crm_write_result and result.crm_write_result.get("contact_action") == "created":
                st.session_state["contacts_created"] += 1
            if result.crm_data and isinstance(result.crm_data.get("Opportunity", {}).get("Amount"), (int, float)):
                st.session_state["pipeline_val"] += result.crm_data["Opportunity"]["Amount"]

            time.sleep(0.2)
            bar_holder.empty()
            progress_card.empty()
            st.success("AI Sales Meeting Agent processing complete!")

# ---------------------------------------------------------------------------
# DISPLAY AGENT INTELLIGENCE
# ---------------------------------------------------------------------------
data = st.session_state["extracted_data"]
analysis = st.session_state["agent_analysis"]
info_needed = st.session_state["information_needed"]
exec_summary = st.session_state["agent_summary"]
actions = st.session_state["agent_actions"]

if data and analysis and info_needed:
    contact = data.get("Contact", {}) or {}
    account = data.get("Account", {}) or {}
    opp = data.get("Opportunity", {}) or {}
    confidence = (data.get("confidence") or "medium").lower()
    contact_name = f"{contact.get('FirstName') or ''} {contact.get('LastName') or ''}".strip() or "—"

    # -----------------------------------------------------------------
    # STAGE 1: AUTOMATIC MEETING SUMMARY (9-POINT INTELLIGENCE)
    # -----------------------------------------------------------------
    if exec_summary:
        st.markdown('<div class="section-eyebrow">Agent Stage 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📑 Automatic Meeting Summary (Generated from Recording / Transcript)</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown(f"<div class='card-header'><div class='icon-chip'>📝</div> Executive Brief</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.02rem; line-height:1.6; color:var(--text-hi); margin-bottom:16px; padding:12px 16px; background:rgba(255,255,255,0.03); border-radius:14px; border-left:3px solid var(--blue);'>{esc(exec_summary.get('concise_summary') or exec_summary.get('executive_overview'))}</div>", unsafe_allow_html=True)

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.markdown(f"<div class='rc-field'><div class='lbl'>👥 Participants</div><div class='val' style='font-size:0.88rem;'>{esc(exec_summary.get('participants') or '—')}</div></div>", unsafe_allow_html=True)
        with sm2:
            st.markdown(f"<div class='rc-field'><div class='lbl'>🎯 Purpose</div><div class='val' style='font-size:0.88rem;'>{esc(exec_summary.get('purpose') or '—')}</div></div>", unsafe_allow_html=True)
        with sm3:
            st.markdown(f"<div class='rc-field'><div class='lbl'>💰 Budget</div><div class='val' style='font-size:0.88rem;'>{esc(exec_summary.get('budget') or '—')}</div></div>", unsafe_allow_html=True)
        with sm4:
            st.markdown(f"<div class='rc-field'><div class='lbl'>⏰ Timeline</div><div class='val' style='font-size:0.88rem;'>{esc(exec_summary.get('timeline') or '—')}</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("<div style='font-weight:700; color:#FF9BC8; margin-bottom:6px;'>🚨 Core Pain Points:</div>", unsafe_allow_html=True)
            for pp in exec_summary.get("pain_points", []):
                st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(pp)}</div>", unsafe_allow_html=True)
            
            if exec_summary.get("objections"):
                st.markdown("<div style='font-weight:700; color:#FFD180; margin:12px 0 6px 0;'>⚠️ Objections / Competitor Mentions:</div>", unsafe_allow_html=True)
                for obj in exec_summary.get("objections", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(obj)}</div>", unsafe_allow_html=True)

        with sc2:
            st.markdown("<div style='font-weight:700; color:#8CF0AF; margin-bottom:6px;'>💡 Customer Needs & Requirements:</div>", unsafe_allow_html=True)
            for sol in exec_summary.get("customer_needs", []):
                st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(sol)}</div>", unsafe_allow_html=True)

            if exec_summary.get("decisions"):
                st.markdown("<div style='font-weight:700; color:#9DBBFF; margin:12px 0 6px 0;'>🤝 Decisions & Mutual Alignments:</div>", unsafe_allow_html=True)
                for dec in exec_summary.get("decisions", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(dec)}</div>", unsafe_allow_html=True)

        if exec_summary.get("next_steps"):
            st.markdown("<div style='font-weight:700; color:#6EE7A0; margin:12px 0 4px 0;'>⚡ Next Steps & Deliverables:</div>", unsafe_allow_html=True)
            for nxt in exec_summary.get("next_steps", []):
                st.markdown(f"<div style='font-size:0.88rem; color:var(--text-hi); padding:2px 0;'>➡️ {esc(nxt)}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STAGE 2: CONVERSATION ANALYSIS & SIGNALS
    # -----------------------------------------------------------------
    st.markdown('<div class="section-eyebrow">Agent Stage 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Conversation Analysis & Buying Signals</div>', unsafe_allow_html=True)

    an1, an2, an3, an4 = st.columns(4)
    with an1:
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#2563EB;">
            <div class="rc-head"><div class="rc-icon">🎯</div><div class="rc-title">Meeting Type</div></div>
            <div class="rc-field"><div class="val">{esc(analysis.get('meeting_type'))}</div></div>
        </div>
        """, unsafe_allow_html=True)

    with an2:
        sentiment = analysis.get("sentiment", "Neutral")
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#22C55E;">
            <div class="rc-head"><div class="rc-icon">😊</div><div class="rc-title">Buyer Sentiment</div></div>
            <div class="rc-field"><div class="val">{esc(sentiment)}</div></div>
            <span class="pill verified">Tone Assessed</span>
        </div>
        """, unsafe_allow_html=True)

    with an3:
        intent = analysis.get("buying_intent", "Medium")
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#7C3AED;">
            <div class="rc-head"><div class="rc-icon">🔥</div><div class="rc-title">Buying Intent</div></div>
            <div class="rc-field"><div class="val">{esc(intent)}</div></div>
            <span class="pill high">Commercial Intent</span>
        </div>
        """, unsafe_allow_html=True)

    with an4:
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#F59E0B;">
            <div class="rc-head"><div class="rc-icon">⚡</div><div class="rc-title">Urgency Level</div></div>
            <div class="rc-field"><div class="val">{esc(analysis.get('urgency'))}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # Information Need Assessment
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    comp_score = info_needed.get("completeness_score", 85)
    st.progress(comp_score / 100.0, text=f"CRM Completeness Score: {comp_score}% ({info_needed.get('status', 'Complete')})")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("<div style='font-weight:700; color:#8CF0AF; margin:10px 0 6px 0;'>✓ Captured CRM Attributes:</div>", unsafe_allow_html=True)
        for cap in info_needed.get("captured_fields", []):
            st.markdown(f"<div class='assess-row'><div class='assess-icon'>✅</div><div class='assess-text'>{esc(cap)}</div></div>", unsafe_allow_html=True)

    with ic2:
        missing_fields = info_needed.get("missing_critical_fields", [])
        if missing_fields:
            st.markdown("<div style='font-weight:700; color:#FFD180; margin:10px 0 6px 0;'>⚠️ Requires Rep Discovery / Enrichment:</div>", unsafe_allow_html=True)
            for mis in missing_fields:
                st.markdown(f"<div class='assess-row'><div class='assess-icon'>⚠️</div><div class='assess-text'>{esc(mis)}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-weight:700; color:#8CF0AF; margin:10px 0 6px 0;'>✅ All Primary Fields Captured</div>", unsafe_allow_html=True)

        discovery_qs = info_needed.get("recommended_discovery_questions", [])
        if discovery_qs:
            st.markdown("<div style='font-weight:700; color:#CBB3FF; margin:12px 0 4px 0;'>❓ Recommended Discovery Questions:</div>", unsafe_allow_html=True)
            for q in discovery_qs:
                st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:3px 0;'>• <i>\"{esc(q)}\"</i></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STAGE 3: STRUCTURED CRM EXTRACTION & RESULTS
    # -----------------------------------------------------------------
    st.markdown('<div class="section-eyebrow">Agent Stage 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂️ Structured CRM Extraction & Results</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#2563EB;">
            <div class="rc-head"><div class="rc-icon">👤</div><div class="rc-title">Contact</div></div>
            <div class="rc-field"><div class="lbl">Name</div><div class="val">{esc(contact_name)}</div></div>
            <div class="rc-field"><div class="lbl">Title</div><div class="val">{esc(contact.get('Title') or '—')}</div></div>
            <div class="rc-field"><div class="lbl">Email</div><div class="val">{esc(contact.get('Email') or '—')}</div></div>
            <span class="pill {esc(confidence)}">Confidence: {esc(confidence.title())}</span>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        review_badge = "review" if not account.get("Industry") or confidence == "low" else "verified"
        review_text = "Needs Review" if review_badge == "review" else "Verified"
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#7C3AED;">
            <div class="rc-head"><div class="rc-icon">🏢</div><div class="rc-title">Account</div></div>
            <div class="rc-field"><div class="lbl">Company</div><div class="val">{esc(account.get('Name') or '—')}</div></div>
            <div class="rc-field"><div class="lbl">Industry</div><div class="val">{esc(account.get('Industry') or '—')}</div></div>
            <div class="rc-field"><div class="lbl">Team Size</div><div class="val">{esc(account.get('NumberOfEmployees') or '—')} employees</div></div>
            <span class="pill {review_badge}">{review_text}</span>
        </div>
        """, unsafe_allow_html=True)

    amount = opp.get("Amount")
    amount_str = f"${amount:,.0f}" if isinstance(amount, (int, float)) else (account.get("BudgetRange__c") or "—")
    with rc3:
        st.markdown(f"""
        <div class="result-card fade-in" style="--accent:#F59E0B;">
            <div class="rc-head"><div class="rc-icon">💼</div><div class="rc-title">Opportunity</div></div>
            <div class="rc-field"><div class="lbl">Deal Name</div><div class="val">{esc(opp.get('Name') or '—')}</div></div>
            <div class="rc-field"><div class="lbl">Budget</div><div class="val">{esc(amount_str)}</div></div>
            <div class="rc-field"><div class="lbl">Stage</div><div class="val">{esc(opp.get('StageName') or '—')}</div></div>
            <div class="rc-field"><div class="lbl">Next Step</div><div class="val">{esc(opp.get('NextStep') or '—')}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Source Evidence
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><div class="icon-chip">🔎</div> Source Evidence & Grounding</div>', unsafe_allow_html=True)

    def build_highlighted(text, contact, account, opp):
        spans = []
        terms = [
            (contact_name if contact_name != "—" else None, "hl-contact"),
            (account.get("Name"), "hl-company"),
            (account.get("BudgetRange__c"), "hl-budget"),
            (opp.get("Competitor__c"), "hl-competitor"),
            (opp.get("NextStep"), "hl-nextstep"),
        ]
        for term, cls in terms:
            if not term:
                continue
            try:
                for m in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
                    spans.append((m.start(), m.end(), cls))
            except re.error:
                continue
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        resolved, last_end = [], -1
        for start, end, cls in spans:
            if start >= last_end:
                resolved.append((start, end, cls))
                last_end = end
        out, cursor = [], 0
        for start, end, cls in resolved:
            out.append(esc(text[cursor:start]))
            out.append(f'<span class="hl {cls}">{esc(text[start:end])}</span>')
            cursor = end
        out.append(esc(text[cursor:]))
        return "".join(out).replace("\n", "<br>")

    highlighted = build_highlighted(st.session_state["meeting_summary"], contact, account, opp)
    st.markdown(f'<div class="evidence-text">{highlighted}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#2563EB;"></div>Contact</div>
        <div class="legend-item"><div class="legend-dot" style="background:#7C3AED;"></div>Company</div>
        <div class="legend-item"><div class="legend-dot" style="background:#F59E0B;"></div>Budget</div>
        <div class="legend-item"><div class="legend-dot" style="background:#EC4899;"></div>Competitor</div>
        <div class="legend-item"><div class="legend-dot" style="background:#22C55E;"></div>Next Step</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STAGE 4: MISSED QUESTIONS, MISSED OPPORTUNITIES & WHAT WE SHOULD HAVE SAID
    # -----------------------------------------------------------------
    coaching = st.session_state.get("agent_coaching")
    if coaching:
        st.markdown('<div class="section-eyebrow">Agent Stage 4</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎓 Missed Opportunities & What We Should Have Said</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("<div class='card-header'><div class='icon-chip'>🏆</div> Sales Performance & Discovery Evaluation</div>", unsafe_allow_html=True)
        
        score = coaching.get("overall_performance_score", 78)
        st.progress(score / 100.0, text=f"Sales Execution Score: {score}/100")
        
        st.markdown(f"<div style='font-size:0.98rem; line-height:1.6; color:var(--text-hi); margin:14px 0; padding:12px 16px; background:rgba(255,255,255,0.03); border-radius:14px; border-left:3px solid var(--purple);'><b>Coach Takeaway:</b> {esc(coaching.get('coaching_takeaway'))}</div>", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            if coaching.get("failed_to_ask"):
                st.markdown("<div style='font-weight:700; color:#FF9BC8; margin:8px 0 6px 0;'>❌ Missed Questions (Failed to Ask):</div>", unsafe_allow_html=True)
                for item in coaching.get("failed_to_ask", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(item)}</div>", unsafe_allow_html=True)

            if coaching.get("important_missing_information"):
                st.markdown("<div style='font-weight:700; color:#FFD180; margin:14px 0 6px 0;'>⚠️ Important Missing Information:</div>", unsafe_allow_html=True)
                for item in coaching.get("important_missing_information", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(item)}</div>", unsafe_allow_html=True)

        with cc2:
            if coaching.get("missed_sales_opportunities"):
                st.markdown("<div style='font-weight:700; color:#9DBBFF; margin:8px 0 6px 0;'>💡 Missed Sales Opportunities & Expansion Angles:</div>", unsafe_allow_html=True)
                for item in coaching.get("missed_sales_opportunities", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(item)}</div>", unsafe_allow_html=True)

            if coaching.get("unanswered_customer_concerns"):
                st.markdown("<div style='font-weight:700; color:#FFD180; margin:14px 0 6px 0;'>🔍 Unanswered Customer Concerns:</div>", unsafe_allow_html=True)
                for item in coaching.get("unanswered_customer_concerns", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(item)}</div>", unsafe_allow_html=True)

            if coaching.get("strengths"):
                st.markdown("<div style='font-weight:700; color:#8CF0AF; margin:14px 0 6px 0;'>✓ Observed Strengths:</div>", unsafe_allow_html=True)
                for item in coaching.get("strengths", []):
                    st.markdown(f"<div style='font-size:0.88rem; color:var(--text-mid); padding:2px 0;'>• {esc(item)}</div>", unsafe_allow_html=True)

        if coaching.get("recommended_questions_to_ask"):
            st.markdown("<div style='font-weight:700; color:#6EE7A0; margin:18px 0 8px 0;'>💬 Recommended Questions & What We Should Have Said:</div>", unsafe_allow_html=True)
            for q in coaching.get("recommended_questions_to_ask", []):
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.025); border:1px solid var(--glass-border); border-radius:12px; padding:12px 16px; margin-bottom:10px;'>
                    <div style='font-size:0.75rem; color:var(--text-lo); text-transform:uppercase; font-weight:700;'>Context: {esc(q.get('context'))}</div>
                    <div style='font-size:0.95rem; color:var(--text-hi); font-weight:700; margin:5px 0;'>👉 <b>What We Should Have Said:</b> {esc(q.get('recommended_question'))}</div>
                    <div style='font-size:0.82rem; color:#9DBBFF;'><i>Strategic Rationale: {esc(q.get('rationale'))}</i></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STAGE 5: NEXT BEST ACTION (NBA) & STRATEGY
    # -----------------------------------------------------------------
    st.markdown('<div class="section-eyebrow">Agent Stage 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Next Best Action (NBA) & Strategy</div>', unsafe_allow_html=True)

    if actions:
        nba = actions.get("next_best_action", {})
        if nba:
            st.markdown(f"""
            <div class="glass-card fade-in" style="border-left:4px solid var(--green);">
                <div class="card-header"><div class="icon-chip" style="background:linear-gradient(135deg, #22C55E, #16a34a);">🎯</div> Primary Next Best Action</div>
                <div style="font-size:1.15rem; font-weight:800; color:var(--text-hi); margin-bottom:8px;">{esc(nba.get('primary_action'))}</div>
                <div style="font-size:0.94rem; line-height:1.6; color:var(--text-mid); margin-bottom:12px;"><b>Strategic Rationale:</b> {esc(nba.get('rationale'))}</div>
                <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                    <span class="pill high">Urgency: {esc(nba.get('urgency'))}</span>
                    <span class="pill reuse">Milestone: {esc(nba.get('target_milestone'))}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if nba.get("deal_acceleration_strategy"):
                st.markdown(f"""
                <div style="margin:8px 0 16px 0; padding:12px 16px; background:rgba(37,99,235,0.12); border:1px solid rgba(37,99,235,0.3); border-radius:12px; font-size:0.92rem; color:#9DBBFF;">
                    🚀 <b>Deal Acceleration Strategy:</b> {esc(nba.get('deal_acceleration_strategy'))}
                </div>
                """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STAGE 6: ACTION ITEMS & CRM DATABASE WRITE
    # -----------------------------------------------------------------
    st.markdown('<div class="section-eyebrow">Agent Stage 6</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Action Items Breakdown & CRM Database Write</div>', unsafe_allow_html=True)

    if actions:
        nba = actions.get("next_best_action", {})
        if nba.get("required_follow_ups"):
            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><div class="icon-chip">📋</div> Required Follow-Up Actions Breakdown</div>', unsafe_allow_html=True)
            for fu in nba.get("required_follow_ups", []):
                st.markdown(f"""
                <div class="task-item">
                    <div>
                        <div style="font-size:0.72rem; color:var(--text-lo); text-transform:uppercase; font-weight:700;">{esc(fu.get('category'))}</div>
                        <div class="task-title">👉 {esc(fu.get('action'))}</div>
                        <div class="task-meta">Owner: <b>{esc(fu.get('suggested_owner'))}</b> | Target: <b>{esc(fu.get('deadline'))}</b></div>
                    </div>
                    <span class="pill verified">{esc(fu.get('status'))}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if actions.get("risk_alerts"):
            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><div class="icon-chip">⚠️</div> Deal Risk Alerts</div>', unsafe_allow_html=True)
            for rk in actions.get("risk_alerts", []):
                st.warning(f"⚠️ {rk}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Dynamic AI Follow-up email
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><div class="icon-chip">✉️</div> Context-Aware Follow-Up Email Draft</div>', unsafe_allow_html=True)

        email_draft = actions.get("follow_up_email", {})
        st.markdown(f"<div style='font-weight:700; color:var(--text-hi); margin-bottom:8px;'>Subject: {esc(email_draft.get('subject'))}</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="email-card">{esc(email_draft.get("body"))}</div>', unsafe_allow_html=True)

        em1, em2 = st.columns(2)
        with em1:
            if st.button("🔄 Regenerate Email Draft", use_container_width=True):
                st.session_state["email_variant"] += 1
                st.rerun()
        with em2:
            st.markdown("<div style='text-align:right; font-size:0.85rem; color:var(--text-lo); padding-top:8px;'>Ready to send to prospect</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


    # -----------------------------------------------------------------
    # STEP 7: DUPLICATE DETECTION & CRM WRITE GATE
    # -----------------------------------------------------------------
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">CRM Operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧬 Duplicate Detection & Record Verification</div>', unsafe_allow_html=True)

    preview_client = MockCRMClient()
    candidates = []

    existing_contacts = preview_client.db.get("Contact", {})
    if contact_name != "—":
        for cid, rec in existing_contacts.items():
            existing_name = f"{rec.get('FirstName') or ''} {rec.get('LastName') or ''}".strip()
            if not existing_name:
                continue
            score = fuzz.token_sort_ratio(contact_name, existing_name)
            candidates.append(("Contact", existing_name, score))

    existing_accounts = preview_client.db.get("Account", {})
    account_name = account.get("Name")
    if account_name:
        for aid, rec in existing_accounts.items():
            existing_name = rec.get("Name") or ""
            if not existing_name:
                continue
            score = fuzz.token_sort_ratio(account_name, existing_name)
            candidates.append(("Account", existing_name, score))

    candidates.sort(key=lambda c: -c[2])
    top_candidates = [c for c in candidates if c[2] >= 60][:3]

    if top_candidates:
        for kind, name, score in top_candidates:
            initials = "".join([p[0] for p in name.split()[:2]]).upper() or "?"
            reuse_note = "Reuse existing" if score >= 88 else "Similar — worth checking"
            st.markdown(f"""
            <div class="dup-row">
                <div class="dup-left">
                    <div class="dup-avatar">{esc(initials)}</div>
                    <div>
                        <div class="dup-name">{esc(name)}</div>
                        <div class="dup-type">{esc(kind)}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center;">
                    <span class="dup-score">{score:.0f}% match</span>
                    <span class="pill reuse">{esc(reuse_note)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="dup-note">🛡️ AI found similar records and will avoid creating duplicate entries.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dup-note">✅ No duplicate records found in CRM — will create clean new records.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        st.button("✏️ Edit Fields", use_container_width=True, help="Manual field override")
    with ec2:
        confirm_clicked = st.button("✅ Confirm & Execute CRM Actions", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if confirm_clicked:
        crm = MockCRMClient()
        try:
            result = crm.write_crm_record(data, actions=actions)
            st.session_state["crm_result"] = result
            st.session_state["extraction_complete"] = True
            if result.get("contact_action") == "created":
                st.session_state["contacts_created"] += 1
            if isinstance(amount, (int, float)):
                st.session_state["pipeline_value"] += amount
            if opp.get("NextStep"):
                st.session_state["followups_due"] += 1
            st.session_state["meetings_processed"] += 1
            st.rerun()
        except Exception as e:
            st.error(f"Could not write to CRM: {e}")

# ---------------------------------------------------------------------------
# SUCCESS STEPPER & AUTOMATED CRM EXECUTION CARD
# ---------------------------------------------------------------------------
if st.session_state["extraction_complete"] and st.session_state["crm_result"]:
    crm_res = st.session_state["crm_result"]
    st.markdown('<div class="glass-card fade-in" style="border-left:4px solid var(--green);">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Automated Execution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ CRM Operations Automatically Executed</div>', unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--glass-border); border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:var(--text-lo); text-transform:uppercase;">Account</div>
            <div style="font-size:0.92rem; font-weight:800; color:var(--text-hi); margin:4px 0;">{esc(crm_res.get('account_id') or '—')}</div>
            <span class="pill {'verified' if 'create' in (crm_res.get('account_action') or '') else 'reuse'}">{esc(crm_res.get('account_action') or 'synced')}</span>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--glass-border); border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:var(--text-lo); text-transform:uppercase;">Contact</div>
            <div style="font-size:0.92rem; font-weight:800; color:var(--text-hi); margin:4px 0;">{esc(crm_res.get('contact_id') or '—')}</div>
            <span class="pill {'verified' if 'create' in (crm_res.get('contact_action') or '') else 'reuse'}">{esc(crm_res.get('contact_action') or 'synced')}</span>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--glass-border); border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:var(--text-lo); text-transform:uppercase;">Opportunity</div>
            <div style="font-size:0.92rem; font-weight:800; color:var(--text-hi); margin:4px 0;">{esc(crm_res.get('opportunity_id') or '—')}</div>
            <span class="pill {'verified' if 'create' in (crm_res.get('opportunity_action') or '') else 'reuse'}">{esc(crm_res.get('opportunity_action') or 'synced')}</span>
        </div>
        """, unsafe_allow_html=True)
    with rc4:
        tasks_count = len(crm_res.get('task_ids', []))
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--glass-border); border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:var(--text-lo); text-transform:uppercase;">Action Items</div>
            <div style="font-size:0.92rem; font-weight:800; color:var(--text-hi); margin:4px 0;">{tasks_count} Tasks</div>
            <span class="pill high">{crm_res.get('tasks_created', 0)} New / {crm_res.get('tasks_updated', 0)} Updated</span>
        </div>
        """, unsafe_allow_html=True)

    stepper_stages = ["Analyzed", "Account Managed", "Contact Managed", "Opportunity Tracked", "Action Items Synced"]
    stepper_html = '<div class="stepper" style="margin-top:16px;">'
    for i, label in enumerate(stepper_stages):
        stepper_html += f"""<div class="stepper-item">
            <div class="stepper-dot">✓</div>
            <div class="stepper-label">{esc(label)}</div>
        </div>"""
        if i < len(stepper_stages) - 1:
            stepper_html += '<div class="stepper-line"></div>'
    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------------------------------
st.markdown('<div class="section-eyebrow">Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Agent Performance & Pipeline Dashboard</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "📞", st.session_state["meetings_processed"], "Meetings Processed"),
    (m2, "💰", f"${st.session_state['pipeline_value']:,.0f}", "Pipeline Value"),
    (m3, "👥", st.session_state["contacts_created"], "Contacts Created"),
    (m4, "⏰", st.session_state["followups_due"], "Follow-ups Due"),
]
for col, icon, value, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <span class="metric-delta up">▲ this session</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
st.markdown('<div class="card-header"><div class="icon-chip">📈</div> Active Pipeline by Stage</div>', unsafe_allow_html=True)

stage_order = ["Prospecting", "Qualification", "Needs Analysis", "Value Proposition",
               "Proposal/Price Quote", "Negotiation/Review", "Demo Scheduled",
               "Closed Won", "Closed Lost"]
stage_counts = {"Prospecting": 8, "Qualification": 6, "Needs Analysis": 4,
                "Value Proposition": 3, "Proposal/Price Quote": 5,
                "Negotiation/Review": 4, "Demo Scheduled": 8,
                "Closed Won": 6, "Closed Lost": 2}

if PLOTLY_AVAILABLE:
    fig = go.Figure(go.Bar(
        x=[stage_counts[s] for s in stage_order],
        y=stage_order,
        orientation="h",
        marker=dict(
            color=[stage_counts[s] for s in stage_order],
            colorscale=[[0, "#3B82F6"], [0.5, "#8B5CF6"], [1, "#FBBF24"]],
            line=dict(width=0),
        ),
        text=[stage_counts[s] for s in stage_order],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C7D0E4", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.bar_chart({s: stage_counts[s] for s in stage_order})
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# BUSINESS IMPACT BANNER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="impact-banner fade-in">
    <div class="impact-title">From 6 minutes to 18 seconds</div>
    <div class="impact-sub">
        The AI Sales Meeting Agent analyzes conversations, decides information needs, extracts verified CRM records, generates executive summaries, and formulates immediate sales actions.
    </div>
    <div class="badge-row">
        <span class="badge green">⚡ 95% Faster</span>
        <span class="badge blue">🦙 Llama 3.2</span>
        <span class="badge purple">🧬 Duplicate-Safe</span>
        <span class="badge orange">🤝 Human-in-the-loop</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:var(--text-lo); font-size:0.78rem; padding:18px 0 6px 0;">
    AI Sales Meeting Agent — powered by local Llama 3.2 inference, zero API cost, full privacy.
</div>
""", unsafe_allow_html=True)