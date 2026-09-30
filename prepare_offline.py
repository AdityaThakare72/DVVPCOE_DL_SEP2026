"""
prepare_offline.py
------------------
Run this ONCE on a machine that HAS internet access.
It downloads every dataset and pretrained weight file used by the LP-IV
Deep Learning practicals and stores them inside this folder:

    datasets/mnist.npz                 (Practical 2)
    datasets/cifar10.npz               (Practical 3 and 6)
    datasets/ecg.csv                   (Practical 4)
    datasets/cbow_corpus.txt           (Practical 5, already included, no download)
    models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5   (Practical 6)

After it finishes, copy the WHOLE folder (notebooks + datasets + models) to
each lab machine (pen drive / shared drive). No internet is needed after that.

Usage (from inside this folder):
    python prepare_offline.py

Only the Python standard library and numpy are used, so it works right after
`pip install -r requirements.txt`.
"""

import os
import sys
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Official source URLs (the same ones Keras / TensorFlow use internally)
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ECG_URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
MOBILENET_FILE = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
MOBILENET_URL = ("https://storage.googleapis.com/tensorflow/keras-applications/"
                 "mobilenet_v2/" + MOBILENET_FILE)


def download(url, dest):
    """Download url to dest with a simple progress indicator. Skips if present."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\r    {done/1e6:7.1f} / {total/1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    os.replace(tmp, dest)
    return dest


def cifar_tar_to_npz(tar_path, npz_path):
    """Convert the official CIFAR-10 python tarball into one compact .npz file."""
    x_train, y_train = [], []
    with tarfile.open(tar_path, "r:gz") as tar:
        def load_batch(name):
            member = tar.getmember(f"cifar-10-batches-py/{name}")
            d = pickle.load(tar.extractfile(member), encoding="bytes")
            # stored as (N, 3072) = 3 colour planes of 32x32 -> convert to (N, 32, 32, 3)
            x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y = np.array(d[b"labels"], dtype=np.uint8)
            return x, y

        for i in range(1, 6):
            x, y = load_batch(f"data_batch_{i}")
            x_train.append(x)
            y_train.append(y)
        x_test, y_test = load_batch("test_batch")

    np.savez_compressed(npz_path,
                        x_train=np.concatenate(x_train), y_train=np.concatenate(y_train),
                        x_test=x_test, y_test=y_test)


def main():
    # 1. MNIST (about 11 MB)
    download(MNIST_URL, DATA_DIR / "mnist.npz")

    # 2. CIFAR-10 (about 163 MB tarball -> converted to cifar10.npz)
    cifar_npz = DATA_DIR / "cifar10.npz"
    if cifar_npz.exists():
        print("[skip] cifar10.npz already exists")
    else:
        tar_path = download(CIFAR_URL, DATA_DIR / "cifar-10-python.tar.gz")
        print("[convert] cifar-10-python.tar.gz -> cifar10.npz (takes a minute)")
        cifar_tar_to_npz(tar_path, cifar_npz)
        tar_path.unlink()  # the tarball is no longer needed

    # 3. ECG5000 (about 5 MB)
    download(ECG_URL, DATA_DIR / "ecg.csv")

    # 4. MobileNetV2 ImageNet weights without the top classifier, 96x96 input (about 9 MB)
    download(MOBILENET_URL, MODEL_DIR / MOBILENET_FILE)

    # 5. CBOW corpus ships with the folder
    if not (DATA_DIR / "cbow_corpus.txt").exists():
        print("[warning] datasets/cbow_corpus.txt is missing - copy it from the original bundle")

    print("\nVerifying files...")
    m = np.load(DATA_DIR / "mnist.npz")
    print("  mnist   :", m["x_train"].shape, m["x_test"].shape)
    c = np.load(cifar_npz)
    print("  cifar10 :", c["x_train"].shape, c["x_test"].shape)
    rows = sum(1 for _ in open(DATA_DIR / "ecg.csv"))
    print("  ecg.csv :", rows, "rows")
    print("  weights :", (MODEL_DIR / MOBILENET_FILE).stat().st_size / 1e6, "MB")
    print("\nAll set. Copy this whole folder to the lab machines, then run check_setup.py there.")


if __name__ == "__main__":
    main()
