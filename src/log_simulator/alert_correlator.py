"""
Alert Correlator

Takes parsed SSH log events and groups them into incidents.
This is the "correlation" step — the same thing a SOC analyst does
when they look at individual alerts and ask "are these related?"

The correlator looks for patterns like:
  - Many failed logins from the same IP in a short time window (brute force)
  - A successful login immediately following failed attempts (possible compromise)
  - Login from an IP that has never been seen before (anomaly)

Each incident gets a severity score and a label, which is what
our ML model will later learn to predict automatically.
"""

import json
from collections import defaultdict
from datetime import datetime


def load_parsed_events(input_file="data/raw/ssh_parsed.json"):
    """Load the structured events from the parser output."""
    with open(input_file, "r") as f:
        return json.load(f)


def group_events_by_ip(events):
    """
    Group all events by source IP address.
    This is the first step in finding brute-force patterns —
    an IP that appears many times is suspicious.
    """
    ip_groups = defaultdict(list)
    for event in events:
        ip_groups[event["source_ip"]].append(event)
    return ip_groups


def analyze_ip_activity(ip, events):
    """
    Analyze all events from a single IP and determine if the
    activity looks like an attack.

    Returns an incident dictionary with:
      - classification (brute_force_attempt, brute_force_success, normal)
      - severity (critical, high, medium, low, info)
      - evidence (what made us flag this)
    """
    failed_count = sum(1 for e in events if e["event_type"] == "failed_login")
    success_count = sum(1 for e in events if e["event_type"] == "successful_login")
    total_count = len(events)
    unique_usernames = set(e["username"] for e in events)

    # ---- BRUTE FORCE DETECTION LOGIC ----

    # Pattern 1: Many failed logins from same IP = brute force attempt
    # Threshold: 10+ failed logins is suspicious, 50+ is almost certainly an attack
    if failed_count >= 50:
        # Check if any succeeded after the failures — that means compromise
        if success_count > 0:
            return {
                "source_ip": ip,
                "classification": "brute_force_success",
                "severity": "critical",
                "total_events": total_count,
                "failed_logins": failed_count,
                "successful_logins": success_count,
                "unique_usernames_tried": len(unique_usernames),
                "usernames_tried": list(unique_usernames),
                "evidence": (
                    f"IP {ip} made {failed_count} failed login attempts "
                    f"followed by {success_count} successful login(s). "
                    f"Tried {len(unique_usernames)} different usernames. "
                    f"This strongly indicates a successful brute-force compromise."
                )
            }
        else:
            return {
                "source_ip": ip,
                "classification": "brute_force_attempt",
                "severity": "high",
                "total_events": total_count,
                "failed_logins": failed_count,
                "successful_logins": 0,
                "unique_usernames_tried": len(unique_usernames),
                "usernames_tried": list(unique_usernames),
                "evidence": (
                    f"IP {ip} made {failed_count} failed login attempts "
                    f"trying {len(unique_usernames)} different usernames. "
                    f"None succeeded, but this is a clear brute-force attack."
                )
            }

    elif failed_count >= 10:
        return {
            "source_ip": ip,
            "classification": "suspicious_activity",
            "severity": "medium",
            "total_events": total_count,
            "failed_logins": failed_count,
            "successful_logins": success_count,
            "unique_usernames_tried": len(unique_usernames),
            "usernames_tried": list(unique_usernames),
            "evidence": (
                f"IP {ip} had {failed_count} failed login attempts. "
                f"Below brute-force threshold but warrants investigation."
            )
        }

    # Pattern 2: Normal activity — few or no failures
    else:
        return {
            "source_ip": ip,
            "classification": "normal",
            "severity": "info",
            "total_events": total_count,
            "failed_logins": failed_count,
            "successful_logins": success_count,
            "unique_usernames_tried": len(unique_usernames),
            "usernames_tried": list(unique_usernames),
            "evidence": f"IP {ip} shows normal login activity."
        }


def correlate_and_report(input_file="data/raw/ssh_parsed.json", output_file="data/raw/incidents.json"):
    """
    Main correlation pipeline:
    1. Load parsed events
    2. Group by IP
    3. Analyze each IP's behavior
    4. Produce incident reports
    5. Save and summarize
    """
    events = load_parsed_events(input_file)
    ip_groups = group_events_by_ip(events)

    incidents = []
    for ip, ip_events in ip_groups.items():
        incident = analyze_ip_activity(ip, ip_events)
        incidents.append(incident)

    # Sort by severity for the report
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    incidents.sort(key=lambda x: severity_order.get(x["severity"], 5))

    # Save full incident report
    with open(output_file, "w") as f:
        json.dump(incidents, f, indent=2)

    # Print summary
    print("=" * 60)
    print("INCIDENT CORRELATION REPORT")
    print("=" * 60)
    print(f"Total events analyzed: {len(events)}")
    print(f"Unique source IPs: {len(ip_groups)}")
    print(f"Incidents generated: {len(incidents)}")
    print()

    # Count by classification
    classifications = defaultdict(int)
    for inc in incidents:
        classifications[inc["classification"]] += 1

    for classification, count in classifications.items():
        print(f"  [{classification.upper()}]: {count}")

    print()
    print("-" * 60)
    print("HIGH/CRITICAL INCIDENTS (require attention):")
    print("-" * 60)

    high_priority = [i for i in incidents if i["severity"] in ("critical", "high")]
    if not high_priority:
        print("  None detected.")
    else:
        for inc in high_priority:
            print(f"\n  [{inc['severity'].upper()}] {inc['classification']}")
            print(f"  Source IP: {inc['source_ip']}")
            print(f"  Failed logins: {inc['failed_logins']}")
            print(f"  Successful logins: {inc['successful_logins']}")
            print(f"  Usernames tried: {', '.join(inc['usernames_tried'])}")
            print(f"  Evidence: {inc['evidence']}")

    print()
    print(f"Full report saved to: {output_file}")

    return incidents


if __name__ == "__main__":
    correlate_and_report()