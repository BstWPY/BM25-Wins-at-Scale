# BM25 vs File-System Agent Failure Analysis

- N=1144: n=150, combined BM25=82.099, FS=86.335, delta=-4.236 pp; BM25-only correct=8, FS-only correct=8; FS search-miss among BM25 wins=6, FS hit-but-wrong=2.
- N=2254: n=150, combined BM25=78.469, FS=83.171, delta=-4.702 pp; BM25-only correct=11, FS-only correct=14; FS search-miss among BM25 wins=8, FS hit-but-wrong=3.
- N=6980: n=150, combined BM25=77.53, FS=74.023, delta=3.507 pp; BM25-only correct=15, FS-only correct=8; FS search-miss among BM25 wins=9, FS hit-but-wrong=6.
- N=42587: n=150, combined BM25=66.601, FS=66.748, delta=-0.147 pp; BM25-only correct=20, FS-only correct=16; FS search-miss among BM25 wins=16, FS hit-but-wrong=4.
- N=131876: n=150, combined BM25=60.008, FS=54.924, delta=5.084 pp; BM25-only correct=28, FS-only correct=12; FS search-miss among BM25 wins=26, FS hit-but-wrong=2.
- N=511959: n=150, combined BM25=56.361, FS=36.695, delta=19.666 pp; BM25-only correct=45, FS-only correct=10; FS search-miss among BM25 wins=42, FS hit-but-wrong=3.
