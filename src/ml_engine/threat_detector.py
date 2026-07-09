"""
ML Threat Detector

Extracts features from parsed SSH log events and trains a machine learning
model to classify IP addresses as normal, suspicious, or malicious.
"""

import json
import random
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from log_simulator.ssh_log_generator import (
    generate_successful_login,
    generate_failed_login,
    generate_brute_force_attack,
    fake,
    LEGITIMATE_USERS,
    COMMON_ATTACK_USERNAMES,
)
from log_simulator.log_parser import parse_log_line


def extract_features_for_ip(events):
    failed = [e for e in events if e["event_type"] == "failed_login"]
    success = [e for e in events if e["event_type"] == "successful_login"]
    total = len(events)

    total_events = total
    failed_count = len(failed)
    success_count = len(success)
    failure_ratio = failed_count / total if total > 0 else 0
    unique_usernames = len(set(e["username"] for e in events))
    attack_usernames_used = len(
        set(e["username"] for e in events) & set(COMMON_ATTACK_USERNAMES)
    )
    has_success_after_failures = 1 if (failed_count > 5 and success_count > 0) else 0
    events_per_username = total / unique_usernames if unique_usernames > 0 else 0

    return {
        "total_events": total_events,
        "failed_count": failed_count,
        "success_count": success_count,
        "failure_ratio": round(failure_ratio, 4),
        "unique_usernames": unique_usernames,
        "attack_usernames_used": attack_usernames_used,
        "has_success_after_failures": has_success_after_failures,
        "events_per_username": round(events_per_username, 2),
    }


def generate_training_data(num_samples=2000):
    print("Generating training data...")
    samples = []
    hostname = "web-server-01"
    base_time = datetime.now()

    for i in range(num_samples):
        behavior = random.choices(
            ["normal", "suspicious", "brute_force"],
            weights=[0.7, 0.15, 0.15],
            k=1
        )[0]

        events = []
        ip = fake.ipv4_public()
        ts = base_time + timedelta(seconds=random.randint(0, 86400))

        if behavior == "normal":
            num_events = random.randint(1, 5)
            user = random.choice(LEGITIMATE_USERS)
            for j in range(num_events):
                event_ts = ts + timedelta(seconds=j * random.randint(60, 3600))
                if random.random() < 0.9:
                    line = generate_successful_login(event_ts, hostname)
                else:
                    line = generate_failed_login(event_ts, hostname, ip, user)
                parsed = parse_log_line(line)
                if parsed:
                    events.append(parsed)
            label = 0

        elif behavior == "suspicious":
            num_fails = random.randint(10, 49)
            for j in range(num_fails):
                event_ts = ts + timedelta(seconds=j * random.uniform(1, 10))
                user = random.choice(COMMON_ATTACK_USERNAMES)
                line = generate_failed_login(event_ts, hostname, ip, user)
                parsed = parse_log_line(line)
                if parsed:
                    events.append(parsed)
            if random.random() < 0.2:
                line = generate_successful_login(
                    ts + timedelta(seconds=num_fails * 5), hostname
                )
                parsed = parse_log_line(line)
                if parsed:
                    events.append(parsed)
            label = 1

        else:
            attack_logs = generate_brute_force_attack(
                ts, hostname, succeeds=(random.random() < 0.3)
            )
            for line in attack_logs:
                parsed = parse_log_line(line)
                if parsed:
                    events.append(parsed)
            label = 2

        if events:
            features = extract_features_for_ip(events)
            features["label"] = label
            features["label_name"] = behavior
            samples.append(features)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    df = pd.DataFrame(samples)
    print(f"Training data generated: {len(df)} samples")
    print(f"  - Normal: {len(df[df['label'] == 0])}")
    print(f"  - Suspicious: {len(df[df['label'] == 1])}")
    print(f"  - Brute force: {len(df[df['label'] == 2])}")

    return df


def train_model(df):
    feature_columns = [
        "total_events", "failed_count", "success_count",
        "failure_ratio", "unique_usernames", "attack_usernames_used",
        "has_success_after_failures", "events_per_username"
    ]
    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    label_names = ["normal", "suspicious", "brute_force"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Normal  Suspicious  BruteForce")
    for i, name in enumerate(label_names):
        print(f"  Actual {name:>12s}  {cm[i][0]:>6d}  {cm[i][1]:>10d}  {cm[i][2]:>10d}")

    print("\nFeature Importance (which features matter most for detection):")
    importances = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<30s} {imp:.4f} {bar}")

    return model, feature_columns


def save_model(model, feature_columns, model_dir="data"):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "threat_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_columns}, f)

    print(f"\nModel saved to: {model_path}")


def predict_from_incidents(model, feature_columns, incidents_file="data/raw/ssh_parsed.json"):
    with open(incidents_file, "r") as f:
        events = json.load(f)

    ip_groups = defaultdict(list)
    for event in events:
        ip_groups[event["source_ip"]].append(event)

    print("\n" + "=" * 60)
    print("ML PREDICTIONS ON ACTUAL LOG DATA")
    print("=" * 60)

    label_names = {0: "NORMAL", 1: "SUSPICIOUS", 2: "BRUTE_FORCE"}
    threats_found = []

    for ip, ip_events in ip_groups.items():
        features = extract_features_for_ip(ip_events)
        feature_vector = pd.DataFrame([features])[feature_columns]

        prediction = model.predict(feature_vector)[0]
        confidence = max(model.predict_proba(feature_vector)[0])

        if prediction > 0:
            threats_found.append({
                "ip": ip,
                "prediction": label_names[prediction],
                "confidence": round(confidence * 100, 1),
                "failed_logins": features["failed_count"],
                "success_logins": features["success_count"],
            })

    if threats_found:
        threats_found.sort(key=lambda x: x["confidence"], reverse=True)
        for t in threats_found:
            print(f"\n  [{t['prediction']}] IP: {t['ip']}")
            print(f"    Confidence: {t['confidence']}%")
            print(f"    Failed logins: {t['failed_logins']}")
            print(f"    Successful logins: {t['success_logins']}")
    else:
        print("  No threats detected.")

    print(f"\nTotal IPs analyzed: {len(ip_groups)}")
    print(f"Threats found: {len(threats_found)}")


if __name__ == "__main__":
    df = generate_training_data(num_samples=2000)
    model, feature_columns = train_model(df)
    save_model(model, feature_columns)
    predict_from_incidents(model, feature_columns)