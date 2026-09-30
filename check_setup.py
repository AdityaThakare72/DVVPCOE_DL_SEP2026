"""
check_setup.py
--------------
Run this on EACH lab machine (no internet needed) before the session:

    python check_setup.py

It confirms that all libraries import, all dataset files are present and
readable, and the MobileNetV2 weights load. Anything marked FAIL must be
fixed before the practical.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

ROOT = Path(__file__).resolve().parent
ok = True


def report(name, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


print("Python:", sys.version.split()[0])

# 1. Libraries
for lib in ["numpy", "pandas", "matplotlib", "sklearn", "tensorflow", "keras"]:
    try:
        mod = __import__(lib)
        report(f"import {lib}", True, getattr(mod, "__version__", ""))
    except Exception as e:
        report(f"import {lib}", False, str(e))

# 2. Dataset files
import numpy as np

files = {
    "datasets/mnist.npz": lambda p: np.load(p)["x_train"].shape,
    "datasets/cifar10.npz": lambda p: np.load(p)["x_train"].shape,
    "datasets/ecg.csv": lambda p: f"{sum(1 for _ in open(p))} rows",
    "datasets/cbow_corpus.txt": lambda p: f"{len(open(p, encoding='utf-8').read().split())} words",
}
for rel, loader in files.items():
    p = ROOT / rel
    try:
        report(rel, p.exists(), str(loader(p)))
    except Exception as e:
        report(rel, False, str(e))

# 3. Pretrained weights for Practical 6
w = ROOT / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
try:
    import keras
    base = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False,
                                          weights=str(w))
    report("MobileNetV2 weights", True, f"{base.count_params():,} params loaded")
except Exception as e:
    report("MobileNetV2 weights", False, str(e))

print("\nREADY" if ok else "\nNOT READY - fix the FAIL lines above")
