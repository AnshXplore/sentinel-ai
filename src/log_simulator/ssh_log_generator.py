"""
SSH Log Generator

Generates realistic SSH authentication log entries in a format similar to
what /var/log/auth.log on a Linux server would produce.

Produces two types of activity:
  1. Normal successful logins from known users
  2. Brute-force attacks: many failed attempts from one IP, sometimes ending in success

This is the raw data our SIEM pipeline will later ingest and analyze.
"""

import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

LEGITIMATE_USERS = ["admin", "priya", "arjun", "root", "deploy", "ubuntu"]

COMMON_ATTACK_USERNAMES = [
    "root", "admin", "administrator", "user", "test",
    "oracle", "postgres", "mysql", "ftp", "guest"
]


def format_log_line(timestamp, hostname, message):
    time_str = timestamp.strftime("%b %d %H:%M:%S")
    return f"{time_str} {hostname} sshd[{random.randint(1000, 9999)}]: {message}"


def generate_successful_login(timestamp, hostname):
    user = random.choice(LEGITIMATE_USERS)
    source_ip = fake.ipv4_public()
    message = f"Accepted password for {user} from {source_ip} port {random.randint(30000, 65000)} ssh2"
    return format_log_line(timestamp, hostname, message)


def generate_failed_login(timestamp, hostname, source_ip, username):
    message = f"Failed password for {username} from {source_ip} port {random.randint(30000, 65000)} ssh2"
    return format_log_line(timestamp, hostname, message)


def generate_brute_force_attack(start_time, hostname, succeeds=False):
    logs = []
    attacker_ip = fake.ipv4_public()
    num_attempts = random.randint(50, 200)
    current_time = start_time

    for _ in range(num_attempts):
        username = random.choice(COMMON_ATTACK_USERNAMES)
        logs.append(generate_failed_login(current_time, hostname, attacker_ip, username))
        current_time += timedelta(seconds=random.uniform(0.1, 2.0))

    if succeeds:
        compromised_user = random.choice(["root", "admin"])
        message = f"Accepted password for {compromised_user} from {attacker_ip} port {random.randint(30000, 65000)} ssh2"
        logs.append(format_log_line(current_time, hostname, message))

    return logs


def generate_mixed_log_stream(num_hours=24, output_file="data/raw/ssh_auth.log"):
    all_logs = []
    hostname = "web-server-01"
    start_time = datetime.now() - timedelta(hours=num_hours)

    normal_login_count = random.randint(5 * num_hours, 15 * num_hours)
    for _ in range(normal_login_count):
        offset = timedelta(seconds=random.randint(0, num_hours * 3600))
        ts = start_time + offset
        log_line = generate_successful_login(ts, hostname)
        all_logs.append((ts, log_line))

    num_attacks = random.randint(2, 4)
    for _ in range(num_attacks):
        offset = timedelta(seconds=random.randint(0, num_hours * 3600 - 600))
        attack_start = start_time + offset
        succeeds = random.random() < 0.3
        attack_logs = generate_brute_force_attack(attack_start, hostname, succeeds=succeeds)
        current_ts = attack_start
        for line in attack_logs:
            all_logs.append((current_ts, line))
            current_ts += timedelta(seconds=random.uniform(0.1, 2.0))

    all_logs.sort(key=lambda x: x[0])

    with open(output_file, "w") as f:
        for _, log_line in all_logs:
            f.write(log_line + "\n")

    print(f"Generated {len(all_logs)} log lines covering {num_hours} hours.")
    print(f"  - ~{normal_login_count} normal successful logins")
    print(f"  - {num_attacks} brute-force attacks injected")
    print(f"  - Written to: {output_file}")


if __name__ == "__main__":
    generate_mixed_log_stream(num_hours=24)