"""Method D: Parameter-Space Coverage Alignment (Validity-SFT -> History-SFT -> Coverage-DPO)
on the frozen HypoSpace 3D benchmark (84 train / 12 validation / 24 test).

All training strictly reads ONLY the 84 train questions. Evaluation uses the unmodified
official Benchmark3D.create_prompt(), official layered voxel output format, m=1 completion
per slot (budget N = |H_O|), zero test-time validator, and official score_predictions.py.
"""
