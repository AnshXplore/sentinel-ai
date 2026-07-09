"""
LLM Investigation Agent

Takes ML-scored incidents and produces human-readable investigation reports.
This is the "brain" of the SOC analyst — it reasons about evidence,
explains its verdict, and recommends actions.

Uses Groq (free, fast) to run Llama/Mixtral models.
"""

import json
import os
import pickle
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml_engine.threat_detector import extract_features_for_ip


def get_groq_client():
    """Initialize the Groq client with API key from .env file."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file.")
        print("Please add your key to the .env file in the project root.")
        sys.exit(1)
    return Groq(api_key=api_key)


def load_ml_model(model_path="data/threat_model.pkl"):
    """Load the trained ML model from disk."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_columns"]


def load_parsed_events(events_file="data/raw/ssh_parsed.json"):
    """Load parsed log events."""
    with open(events_file, "r") as f:
        return json.load(f)


def build_incident_context(ip, events, ml_prediction, ml_confidence, features):
    """
    Build a detailed context string about an incident that the LLM
    can reason about. This is the 'evidence package' we hand to the agent.
    """
    failed_events = [e for e in events if e["event_type"] == "failed_login"]
    success_events = [e for e in events if e["event_type"] == "successful_login"]

    # Build timeline of key events
    timeline = []
    for e in events[:10]:  # First 10 events
        timeline.append(f"  {e['timestamp']} - {e['event_type']} - user: {e['username']}")
    if len(events) > 10:
        timeline.append(f"  ... ({len(events) - 10} more events)")
    # Last 5 events if there are many
    if len(events) > 15:
        for e in events[-5:]:
            timeline.append(f"  {e['timestamp']} - {e['event_type']} - user: {e['username']}")

    timeline_str = "\n".join(timeline)

    context = f"""
INCIDENT INVESTIGATION BRIEF
=============================
Source IP: {ip}
Total Events: {len(events)}
Failed Login Attempts: {len(failed_events)}
Successful Logins: {len(success_events)}
Unique Usernames Attempted: {len(set(e['username'] for e in events))}
Usernames Tried: {', '.join(set(e['username'] for e in events))}
Target Host: {events[0]['hostname'] if events else 'unknown'}

ML MODEL ASSESSMENT:
  Prediction: {ml_prediction}
  Confidence: {ml_confidence}%

EXTRACTED FEATURES:
  Failed Count: {features['failed_count']}
  Success Count: {features['success_count']}
  Failure Ratio: {features['failure_ratio']}
  Attack Usernames Used: {features['attack_usernames_used']}
  Events Per Username: {features['events_per_username']}
  Success After Failures: {features['has_success_after_failures']}

EVENT TIMELINE (sample):
{timeline_str}
"""
    return context


def investigate_incident(client, incident_context):
    """
    Send the incident context to the LLM and get an investigation report.
    The LLM acts as a Tier-1 SOC analyst, reasoning about the evidence.
    """
    system_prompt = """You are an expert SOC (Security Operations Center) analyst 
investigating security incidents. You receive incident data including log events, 
ML model predictions, and extracted features.

Your job is to:
1. Analyze the evidence presented
2. Determine if this is a genuine threat or false positive
3. Assess the severity (Critical, High, Medium, Low, Info)
4. Explain your reasoning step by step, citing specific evidence
5. Recommend immediate actions
6. Assess business impact

Be specific and cite actual numbers from the data. Do not be vague.
Format your response as a structured investigation report."""

    user_prompt = f"""Investigate this security incident and provide your analysis:

{incident_context}

Provide your investigation report with these sections:
1. VERDICT (Threat/False Positive/Inconclusive)
2. SEVERITY (Critical/High/Medium/Low/Info)
3. ATTACK TYPE (if applicable)
4. ANALYSIS (step-by-step reasoning citing evidence)
5. INDICATORS OF COMPROMISE
6. RECOMMENDED ACTIONS
7. BUSINESS IMPACT ASSESSMENT"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: LLM investigation failed: {str(e)}"


def run_full_investigation(events_file="data/raw/ssh_parsed.json"):
    """
    Full investigation pipeline:
    1. Load parsed events
    2. Load trained ML model
    3. Score each IP
    4. For flagged IPs, send to LLM for deep investigation
    5. Produce final report
    """
    print("=" * 60)
    print("SENTINEL-AI: AUTOMATED INCIDENT INVESTIGATION")
    print("=" * 60)

    # Initialize
    print("\n[1/4] Loading ML model...")
    model, feature_columns = load_ml_model()
    print("  Model loaded successfully.")

    print("\n[2/4] Loading parsed events...")
    events = load_parsed_events(events_file)
    print(f"  Loaded {len(events)} events.")

    print("\n[3/4] Running ML threat detection...")
    # Group by IP
    ip_groups = defaultdict(list)
    for event in events:
        ip_groups[event["source_ip"]].append(event)

    label_names = {0: "NORMAL", 1: "SUSPICIOUS", 2: "BRUTE_FORCE"}
    flagged_incidents = []

    for ip, ip_events in ip_groups.items():
        features = extract_features_for_ip(ip_events)
        feature_vector = pd.DataFrame([features])[feature_columns]
        prediction = model.predict(feature_vector)[0]
        confidence = round(max(model.predict_proba(feature_vector)[0]) * 100, 1)

        if prediction > 0:  # Not normal
            flagged_incidents.append({
                "ip": ip,
                "events": ip_events,
                "prediction": label_names[prediction],
                "confidence": confidence,
                "features": features
            })

    print(f"  Scanned {len(ip_groups)} IPs.")
    print(f"  Flagged {len(flagged_incidents)} incidents for investigation.")

    if not flagged_incidents:
        print("\n  No threats detected. All clear.")
        return

    print(f"\n[4/4] LLM agent investigating {len(flagged_incidents)} incidents...")
    client = get_groq_client()

    all_reports = []

    for i, incident in enumerate(flagged_incidents):
        ip = incident["ip"]
        print(f"\n  Investigating incident {i+1}/{len(flagged_incidents)}: {ip}...")

        # Build context for LLM
        context = build_incident_context(
            ip=ip,
            events=incident["events"],
            ml_prediction=incident["prediction"],
            ml_confidence=incident["confidence"],
            features=incident["features"]
        )

        # Send to LLM for investigation
        report = investigate_incident(client, context)

        # Display report
        print("\n" + "=" * 60)
        print(f"INVESTIGATION REPORT - {ip}")
        print(f"ML Prediction: {incident['prediction']} ({incident['confidence']}%)")
        print("=" * 60)
        print(report)

        all_reports.append({
            "ip": ip,
            "ml_prediction": incident["prediction"],
            "ml_confidence": incident["confidence"],
            "features": incident["features"],
            "llm_report": report
        })

    # Save all reports
    output_file = "data/raw/investigation_reports.json"
    with open(output_file, "w") as f:
        json.dump(all_reports, f, indent=2)

    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print(f"Total incidents investigated: {len(all_reports)}")
    print(f"Full reports saved to: {output_file}")


if __name__ == "__main__":
    run_full_investigation()