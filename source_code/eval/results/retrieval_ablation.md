# Retrieval ablation (30 answerable questions, top-5)

Page-Hit = a retrieved chunk comes from the gold page (lenient). Evidence = a retrieved chunk contains the exact evidence sentence (strict).

| Configuration | #chunks | Page-Hit@1 | Page-Hit@5 | Page-MRR | Evidence@5 | Evidence-MRR | Evidence missed |
|---|---|---|---|---|---|---|---|
| R0 baseline (1000/200, L2, no prefix, vector) | 56 | 0.70 | 0.93 | 0.81 | 0.87 | 0.72 | Q16, Q17, Q20, Q31 |
| R1 + nomic task prefixes | 56 | 0.70 | 0.93 | 0.79 | 0.83 | 0.74 | Q13, Q16, Q17, Q20, Q31 |
| R2 + cosine distance | 56 | 0.70 | 0.93 | 0.79 | 0.83 | 0.74 | Q13, Q16, Q17, Q20, Q31 |
| R3 + structure-aware chunking (1000/150) | 58 | 0.73 | 1.00 | 0.83 | 0.87 | 0.75 | Q13, Q16, Q17, Q31 |
| R3b + follow-up query expansion (chat history) | 58 | 0.80 | 1.00 | 0.88 | 0.90 | 0.80 | Q13, Q16, Q17 |
| R3c + merge tiny chunks (<150 chars) | 57 | 0.80 | 1.00 | 0.88 | 0.90 | 0.80 | Q13, Q16, Q17 |
| R4 + hybrid BM25+vector (chunk 1000) | 57 | 0.77 | 0.97 | 0.86 | 0.97 | 0.80 | Q17 |
| R5 final system: hybrid, chunk 800/150 | 69 | 0.70 | 1.00 | 0.83 | 1.00 | 0.79 | - |
| R6 final, vector only | 69 | 0.77 | 0.97 | 0.85 | 0.90 | 0.80 | Q13, Q20, Q23 |
| R7 final, BM25 only | 69 | 0.77 | 1.00 | 0.86 | 0.93 | 0.83 | Q06, Q17 |
| R8 final, chunk size 500 | 100 | 0.80 | 1.00 | 0.89 | 0.93 | 0.71 | Q06, Q17 |
| R9 final, chunk size 1500 | 38 | 0.77 | 0.97 | 0.84 | 0.97 | 0.81 | Q17 |
