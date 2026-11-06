"""Verbatim copies of the v4 LLM prompt templates and response parsers.

Copied from ``research/llm_classification_zero_shot_v4.py`` and
``research/llm_classification_few_shot_v4.py`` (both untouched) so the
revision scripts can build prompts per row without running those scripts.
``check_prompts_v4.py`` asserts these render byte-identical to the originals.

Note the two originals differ in small ways that are kept on purpose:
- zero-shot joins feature lines with "\\n" (no trailing newline); few-shot
  appends "\\n" after every line.
- zero-shot parses only the first line of the reply after stripping quotes;
  few-shot searches the whole reply.
Both fall back to "Benign" when no class name is found.
"""

label_names = {
    0: "Benign",
    1: "Bot",
    2: "Brute Force",
    3: "DoS/DDoS",
    4: "Infiltration",
    5: "PortScan",
    6: "Web Attack",
}

reverse_labels = {v: k for k, v in label_names.items()}

selected_features = [
    "Protocol",
    "Dst Port",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Size Avg",
    "ACK Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Active Mean",
    "Idle Mean",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Pkt Len Max",
    "Pkt Len Min",
    "Down/Up Ratio",
]

knowledge = """
You are an expert cybersecurity analyst specializing in Network Intrusion Detection.

Attack Categories

Benign
- Normal user traffic.
- Balanced communication.
- Typical web or application usage.

Bot
- Malware-controlled host.
- Automated repetitive communication.
- Often shows periodic behaviour.

Brute Force
- Multiple authentication attempts.
- Often targets login services.
- Numerous short connections.

DoS/DDoS
- Extremely high traffic volume.
- Large packet counts.
- High packet rate.
- Victim receives overwhelming traffic.

Infiltration
- Unauthorized internal activity.
- Usually subtle.
- Longer communication sessions.

PortScan
- Sequential probing of many ports.
- Small packets.
- Numerous connection attempts.

Web Attack
- HTTP/HTTPS based attacks.
- SQL Injection.
- XSS.
- Command Injection.

Always compare the traffic against ALL categories before deciding.

Return ONLY ONE class name.

Valid outputs:

Benign
Bot
Brute Force
DoS/DDoS
Infiltration
PortScan
Web Attack
"""


# ==================================================
# ZERO-SHOT
# ==================================================

def zero_shot_prompt(row) -> str:
    traffic = "\n".join(
        [f"{col}: {row[col]}" for col in selected_features]
    )

    prompt = f"""
You are a Network Intrusion Detection System.

Classify this network traffic into exactly ONE class:

Benign
Bot
Brute Force
DoS/DDoS
Infiltration
PortScan
Web Attack

Traffic:
{traffic}

Respond with ONLY ONE class name.
"""
    return prompt


def parse_zero_shot(content: str) -> tuple[str, bool]:
    """Returns (class name, matched). Unmatched -> ("Benign", False)."""
    prediction = (
        content
        .strip()
        .replace('"', "")
        .split("\n")[0]
        .strip()
    )
    for label in reverse_labels:
        if label.lower() in prediction.lower():
            return label, True
    return "Benign", False


# ==================================================
# FEW-SHOT
# ==================================================

def render_example(row) -> str:
    return "".join(f"{feature}: {row[feature]}\n" for feature in selected_features)


def reference_block(examples: dict) -> str:
    """``examples``: {encoded label: rendered example text}."""
    block = ""
    for label in sorted(examples):
        block += f"\nReference Example ({label_names[label]})\n"
        block += examples[label]
        block += "\n"
    return block


def few_shot_prompt(row, ref_block: str) -> str:
    traffic = render_example(row)

    prompt = f"""
{knowledge}

==========================
Representative Examples
==========================

{ref_block}

==========================
Traffic to Classify
==========================

{traffic}

Compare the traffic characteristics with the attack definitions and representative examples.

Determine the single best matching class.

Output ONLY the class name.
"""
    return prompt


def parse_few_shot(content: str) -> tuple[str, bool]:
    """Returns (class name, matched). Unmatched -> ("Benign", False)."""
    answer = content.strip()
    for cls in reverse_labels:
        if cls.lower() in answer.lower():
            return cls, True
    return "Benign", False
