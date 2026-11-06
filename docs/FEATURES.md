# Feature lists

Names follow the canonical (CSE-CIC-IDS2018) convention; see `column_mapping_2017_2018.csv` for the CIC-IDS2017 names.

## 70 model features (in column order)

1. Dst Port
2. Tot Fwd Pkts
3. Tot Bwd Pkts
4. TotLen Fwd Pkts
5. TotLen Bwd Pkts
6. Fwd Pkt Len Max
7. Fwd Pkt Len Min
8. Fwd Pkt Len Mean
9. Fwd Pkt Len Std
10. Bwd Pkt Len Max
11. Bwd Pkt Len Min
12. Bwd Pkt Len Mean
13. Bwd Pkt Len Std
14. Flow Byts/s
15. Flow Pkts/s
16. Fwd IAT Tot
17. Bwd IAT Tot
18. Fwd Header Len
19. Bwd Header Len
20. Fwd Pkts/s
21. Bwd Pkts/s
22. Pkt Len Min
23. Pkt Len Max
24. Pkt Len Mean
25. Pkt Len Std
26. Pkt Len Var
27. FIN Flag Cnt
28. SYN Flag Cnt
29. RST Flag Cnt
30. PSH Flag Cnt
31. ACK Flag Cnt
32. URG Flag Cnt
33. ECE Flag Cnt
34. Pkt Size Avg
35. Fwd Seg Size Avg
36. Bwd Seg Size Avg
37. Subflow Fwd Pkts
38. Subflow Fwd Byts
39. Subflow Bwd Pkts
40. Subflow Bwd Byts
41. Init Fwd Win Byts
42. Init Bwd Win Byts
43. Fwd Act Data Pkts
44. Fwd Seg Size Min
45. Flow Duration
46. Flow IAT Mean
47. Flow IAT Std
48. Flow IAT Max
49. Flow IAT Min
50. Fwd IAT Mean
51. Fwd IAT Std
52. Fwd IAT Max
53. Fwd IAT Min
54. Bwd IAT Mean
55. Bwd IAT Std
56. Bwd IAT Max
57. Bwd IAT Min
58. Fwd PSH Flags
59. Fwd URG Flags
60. CWE Flag Count
61. Down/Up Ratio
62. Active Mean
63. Active Std
64. Active Max
65. Active Min
66. Idle Mean
67. Idle Std
68. Idle Max
69. Idle Min
70. Protocol

## 22 LLM prompt features (in prompt order)

1. Protocol
2. Dst Port
3. Flow Duration
4. Tot Fwd Pkts
5. Tot Bwd Pkts
6. Flow Byts/s
7. Flow Pkts/s
8. Pkt Len Mean
9. Pkt Len Std
10. Pkt Size Avg
11. ACK Flag Cnt
12. SYN Flag Cnt
13. RST Flag Cnt
14. Init Fwd Win Byts
15. Init Bwd Win Byts
16. Active Mean
17. Idle Mean
18. Fwd Pkt Len Mean
19. Bwd Pkt Len Mean
20. Pkt Len Max
21. Pkt Len Min
22. Down/Up Ratio

## 50 features (zero-shot feature-count ablation, in prompt order)

1. Protocol
2. Dst Port
3. Flow Duration
4. Tot Fwd Pkts
5. Tot Bwd Pkts
6. Flow Byts/s
7. Flow Pkts/s
8. Pkt Len Mean
9. Pkt Len Std
10. Pkt Size Avg
11. ACK Flag Cnt
12. SYN Flag Cnt
13. RST Flag Cnt
14. Init Fwd Win Byts
15. Init Bwd Win Byts
16. Active Mean
17. Idle Mean
18. Fwd Pkt Len Mean
19. Bwd Pkt Len Mean
20. Pkt Len Max
21. Pkt Len Min
22. Down/Up Ratio
23. Flow IAT Mean
24. Flow IAT Std
25. Fwd IAT Mean
26. Fwd IAT Std
27. Fwd IAT Max
28. Fwd IAT Min
29. Bwd IAT Mean
30. Bwd IAT Std
31. Bwd IAT Max
32. Bwd IAT Min
33. Bwd Pkt Len Std
34. Bwd Pkt Len Max
35. Bwd Pkt Len Min
36. FIN Flag Cnt
37. PSH Flag Cnt
38. URG Flag Cnt
39. Fwd Header Len
40. Bwd Header Len
41. Fwd Seg Size Avg
42. Bwd Seg Size Avg
43. Subflow Fwd Pkts
44. Subflow Fwd Byts
45. Subflow Bwd Pkts
46. Subflow Bwd Byts
47. Active Std
48. Active Max
49. Idle Std
50. Idle Max
