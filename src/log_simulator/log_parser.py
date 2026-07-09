"""
SSH Log Parser

Reads raw syslog-format SSH log lines and extracts structured fields.
Converts unstructured text into a clean table (list of dictionaries)
that can be used for analysis, ML training, or feeding to an LLM agent.

This is the "normalize" step — the same thing a real SIEM does when
it receives raw logs from different sources and converts them into
a common schema.
"""

import re
import json
from datetime import datetime


# Regular expressions to match the two types of SSH log lines we generate.
# regex is pattern-matching for text — think of it as a very precise search.

# Matches: "Jul 08 18:05:33 web-server-01 sshd[2847]: Failed password for root from 89.44.12.7 port 38291 ssh2"
FAILED_LOGIN_PATTERN = re.compile(
    r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"  # timestamp
    r"(\S+)\s+"                                     # hostname
    r"sshd\[(\d+)\]:\s+"                            # sshd[pid]
    r"Failed password for\s+"                       # event marker
    r"(\S+)\s+"                                     # username
    r"from\s+(\S+)\s+"                              # source IP
    r"port\s+(\d+)"                                 # port
)

# Matches: "Jul 08 18:04:12 web-server-01 sshd[4821]: Accepted password for priya from 142.58.3.201 port 42015 ssh2"
ACCEPTED_LOGIN_PATTERN = re.compile(
    r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"  # timestamp
    r"(\S+)\s+"                                     # hostname
    r"sshd\[(\d+)\]:\s+"                            # sshd[pid]
    r"Accepted password for\s+"                     # event marker
    r"(\S+)\s+"                                     # username
    r"from\s+(\S+)\s+"                              # source IP
    r"port\s+(\d+)"                                 # port
)


def parse_log_line(line):
    """
    Parse a single SSH log line into a structured dictionary.
    Returns None if the line doesn't match any known pattern.
    """
    line = line.strip()
    if not line:
        return None

    # Try matching against failed login pattern first
    match = FAILED_LOGIN_PATTERN.match(line)
    if match:
        return {
            "timestamp": match.group(1),
            "hostname": match.group(2),
            "pid": int(match.group(3)),
            "event_type": "failed_login",
            "username": match.group(4),
            "source_ip": match.group(5),
            "port": int(match.group(6))
        }

    # Try matching against accepted login pattern
    match = ACCEPTED_LOGIN_PATTERN.match(line)
    if match:
        return {
            "timestamp": match.group(1),
            "hostname": match.group(2),
            "pid": int(match.group(3)),
            "event_type": "successful_login",
            "username": match.group(4),
            "source_ip": match.group(5),
            "port": int(match.group(6))
        }

    # Line didn't match any known pattern
    return None


def parse_log_file(input_file="data/raw/ssh_auth.log", output_file="data/raw/ssh_parsed.json"):
    """
    Read the raw log file, parse every line, and save the structured
    output as JSON. Also prints a summary of what was parsed.
    """
    parsed_events = []
    unparsed_count = 0

    with open(input_file, "r") as f:
        for line in f:
            result = parse_log_line(line)
            if result:
                parsed_events.append(result)
            else:
                unparsed_count += 1

    # Save as JSON for easy consumption by other scripts
    with open(output_file, "w") as f:
        json.dump(parsed_events, f, indent=2)

    # Print summary
    total = len(parsed_events)
    failed = sum(1 for e in parsed_events if e["event_type"] == "failed_login")
    success = sum(1 for e in parsed_events if e["event_type"] == "successful_login")
    unique_ips = len(set(e["source_ip"] for e in parsed_events))
    unique_users = len(set(e["username"] for e in parsed_events))

    print(f"Parsed {total} events from {input_file}")
    print(f"  - {failed} failed logins")
    print(f"  - {success} successful logins")
    print(f"  - {unique_ips} unique source IPs")
    print(f"  - {unique_users} unique usernames")
    print(f"  - {unparsed_count} lines could not be parsed")
    print(f"  - Structured output saved to: {output_file}")

    # Show a sample of what the parsed data looks like
    print(f"\nSample parsed event:")
    print(json.dumps(parsed_events[0], indent=2))

    return parsed_events


if __name__ == "__main__":
    parse_log_file()