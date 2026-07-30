# Agent + BM25 ranked-search control

Fixed stratified sample: 150 identical question IDs at each scale. The first Agent search is audited to equal Native BM25's ordered top-5 on every question.

| N | Method | Combined | Correct | Complete | Doc recall | Any-gold hit | Mean calls | Mean tokens/q |
|---:|:--|--:|--:|--:|--:|--:|--:|--:|
| 1,144 | Native BM25 | 81.32 | 90.00 | 85.39 | 92.85 | 95.74 | 1.00 | 5,777 |
| 1,144 | File-System Agent | 87.05 | 90.67 | 90.23 | 91.31 | 95.04 | 8.87 | 129,685 |
| 1,144 | Agent + BM25 | 90.08 | 94.67 | 91.47 | 94.15 | 96.45 | 3.17 | 35,007 |
| 511,959 | Native BM25 | 54.82 | 62.67 | 61.48 | 65.56 | 71.63 | 1.00 | 5,763 |
| 511,959 | File-System Agent | 36.86 | 39.33 | 44.50 | 36.82 | 39.01 | 36.12 | 895,284 |
| 511,959 | Agent + BM25 | 69.38 | 74.00 | 73.63 | 72.42 | 78.01 | 5.79 | 100,942 |

- N=1,144, Agent+BM25 − Native BM25: combined +8.77 pp, paired bootstrap 95% CI [+4.55, +13.09]; correctness +4.67 pp, McNemar p=0.06543.

- N=1,144, Agent+BM25 − File-System Agent: combined +3.04 pp, paired bootstrap 95% CI [-1.71, +7.88]; correctness +4.00 pp, McNemar p=0.1796.

- N=1,144, Native BM25 − File-System Agent: combined -5.73 pp, paired bootstrap 95% CI [-11.57, +0.15]; correctness -0.67 pp, McNemar p=1.

- N=511,959, Agent+BM25 − Native BM25: combined +14.56 pp, paired bootstrap 95% CI [+9.22, +20.12]; correctness +11.33 pp, McNemar p=0.000488.

- N=511,959, Agent+BM25 − File-System Agent: combined +32.52 pp, paired bootstrap 95% CI [+24.07, +40.90]; correctness +34.67 pp, McNemar p=0.

- N=511,959, Native BM25 − File-System Agent: combined +17.97 pp, paired bootstrap 95% CI [+9.68, +26.14]; correctness +23.33 pp, McNemar p=1e-06.


Cost scope: Agent+BM25 charges every attempted row, including failed attempts that were retried; File-System Agent uses exact per-row usage on the same fixed sample. Native BM25 token/q is the full-500 aggregate divided by 500 and is only a descriptive cost reference.
