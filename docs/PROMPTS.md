# Prompt templates (verbatim)

Generated from `research/revision/prompts_v4.py`, which `check_prompts_v4.py` verifies is byte-identical to the original scripts (`llm_classification_{zero,few}_shot_v4.py`). `<Feature>` / `<value>` mark substituted values. Ollama options: temperature 0, num_predict 60; num_ctx default (4096) except the 3-example prompt (8192).

Parsing: zero-shot uses the first response line (quotes stripped), few-shot the whole response; the first class name found by case-insensitive substring match is the prediction, otherwise Benign.

## Zero-shot

```text

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
Protocol: <Protocol>
Dst Port: <Dst Port>
Flow Duration: <Flow Duration>
Tot Fwd Pkts: <Tot Fwd Pkts>
Tot Bwd Pkts: <Tot Bwd Pkts>
Flow Byts/s: <Flow Byts/s>
Flow Pkts/s: <Flow Pkts/s>
Pkt Len Mean: <Pkt Len Mean>
Pkt Len Std: <Pkt Len Std>
Pkt Size Avg: <Pkt Size Avg>
ACK Flag Cnt: <ACK Flag Cnt>
SYN Flag Cnt: <SYN Flag Cnt>
RST Flag Cnt: <RST Flag Cnt>
Init Fwd Win Byts: <Init Fwd Win Byts>
Init Bwd Win Byts: <Init Bwd Win Byts>
Active Mean: <Active Mean>
Idle Mean: <Idle Mean>
Fwd Pkt Len Mean: <Fwd Pkt Len Mean>
Bwd Pkt Len Mean: <Bwd Pkt Len Mean>
Pkt Len Max: <Pkt Len Max>
Pkt Len Min: <Pkt Len Min>
Down/Up Ratio: <Down/Up Ratio>

Respond with ONLY ONE class name.

```

## Few-shot (one example per class)

The reference block contains one real (pre-SMOTE) training row per class, in class order. In the dynamic variant (E4) the same template is used with per-flow retrieved examples; the 3-example variant (E8) numbers them `Reference Example 1..3 (<class>)`.

```text


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


==========================
Representative Examples
==========================


Reference Example (Benign)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>


Reference Example (Bot)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>


Reference Example (Brute Force)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>


Reference Example (DoS/DDoS)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>


Reference Example (Infiltration)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>


Reference Example (PortScan)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>


Reference Example (Web Attack)
Protocol: <value>
Dst Port: <value>
Flow Duration: <value>
Tot Fwd Pkts: <value>
Tot Bwd Pkts: <value>
Flow Byts/s: <value>
Flow Pkts/s: <value>
Pkt Len Mean: <value>
Pkt Len Std: <value>
Pkt Size Avg: <value>
ACK Flag Cnt: <value>
SYN Flag Cnt: <value>
RST Flag Cnt: <value>
Init Fwd Win Byts: <value>
Init Bwd Win Byts: <value>
Active Mean: <value>
Idle Mean: <value>
Fwd Pkt Len Mean: <value>
Bwd Pkt Len Mean: <value>
Pkt Len Max: <value>
Pkt Len Min: <value>
Down/Up Ratio: <value>



==========================
Traffic to Classify
==========================

Protocol: <Protocol>
Dst Port: <Dst Port>
Flow Duration: <Flow Duration>
Tot Fwd Pkts: <Tot Fwd Pkts>
Tot Bwd Pkts: <Tot Bwd Pkts>
Flow Byts/s: <Flow Byts/s>
Flow Pkts/s: <Flow Pkts/s>
Pkt Len Mean: <Pkt Len Mean>
Pkt Len Std: <Pkt Len Std>
Pkt Size Avg: <Pkt Size Avg>
ACK Flag Cnt: <ACK Flag Cnt>
SYN Flag Cnt: <SYN Flag Cnt>
RST Flag Cnt: <RST Flag Cnt>
Init Fwd Win Byts: <Init Fwd Win Byts>
Init Bwd Win Byts: <Init Bwd Win Byts>
Active Mean: <Active Mean>
Idle Mean: <Idle Mean>
Fwd Pkt Len Mean: <Fwd Pkt Len Mean>
Bwd Pkt Len Mean: <Bwd Pkt Len Mean>
Pkt Len Max: <Pkt Len Max>
Pkt Len Min: <Pkt Len Min>
Down/Up Ratio: <Down/Up Ratio>


Compare the traffic characteristics with the attack definitions and representative examples.

Determine the single best matching class.

Output ONLY the class name.

```
