From 40aa76e48ca9d0ecd8597a03a4ceb154dee78dbb Mon Sep 17 00:00:00 2001
From: AdityaThakare72 <AdityaThakare72@users.noreply.github.com>
Date: Wed, 30 Sep 2026 08:25:41 +0000
Subject: [PATCH] readme -> setup + fork guide for students, gitignore => env
 and downloaded data

---
 .gitignore |  15 ++++
 README.md  | 204 +++++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 219 insertions(+)
 create mode 100644 README.md

diff --git a/.gitignore b/.gitignore
index 94548c2..412a0b6 100644
--- a/.gitignore
+++ b/.gitignore
@@ -9,3 +9,18 @@ mnist.npz
 models
 .ipynb_checkpoints
 chess.jpg
+
+# python env
+dl_env/
+__pycache__/
+
+# downloaded data (prepare_offline.py gets these again)
+datasets/*.npz
+datasets/*.tar.gz
+datasets/*.part
+
+# files the notebooks write while running
+*.keras
+p5_vectors.tsv
+p5_metadata.tsv
+executed/
diff --git a/README.md b/README.md
new file mode 100644
index 0000000..ffa195e
--- /dev/null
+++ b/README.md
@@ -0,0 +1,204 @@
+# DVVPCOE Deep Learning - Sep 2026
+
+Deep learning practicals for SPPU BE IT (2019 pattern), **414447: Lab Practice IV**, done in the Sep 2026 session at DVVP COE.
+
+Everything here runs in **Jupyter Notebook on Windows with Python 3.12**, using **TensorFlow / Keras only**, and works **without internet** once the setup below is done. Each notebook follows the syllabus steps as its headings, with comments explaining what each cell does and why, plus small "inspect" cells that print shapes, weights and outputs so you can see what is happening inside.
+
+## What's inside
+
+| # | Practical | Notebook | Dataset |
+|---|---|---|---|
+| 1 | Study of DL packages: TensorFlow, Keras, Theano, PyTorch | slide deck (no code) | - |
+| 2 | Feedforward neural network with Keras and TensorFlow (SGD) | `Practical_2_Feedforward_NN_MNIST.ipynb` | MNIST (CIFAR-10 optional) |
+| 3 | Image classification model in 4 stages (CNN) | `Practical_3_Image_Classification_CNN_CIFAR10.ipynb` | CIFAR-10 |
+| 4 | Anomaly detection using an autoencoder | `Practical_4_Autoencoder_Anomaly_Detection_ECG.ipynb` | ECG5000 |
+| 5 | Continuous Bag of Words (CBOW) model | `Practical_5_CBOW_Word_Embeddings.ipynb` | `datasets/cbow_corpus.txt` (included) |
+| 6 | Object recognition using transfer learning of CNN architectures | `Practical_6_Transfer_Learning_MobileNetV2.ipynb` | CIFAR-10 + MobileNetV2 weights |
+
+Other files:
+
+- `requirements.txt` - exact library versions (TensorFlow 2.19.1, Keras 3.15.1, ...)
+- `prepare_offline.py` - downloads all datasets and pretrained weights into `datasets/` and `models/` (run once, needs internet)
+- `check_setup.py` - checks libraries, datasets and weights; must print `READY` (works offline)
+- `README.txt` - notes for setting up offline lab machines (installing from downloaded wheels)
+
+Class notebooks (used during the session, not part of the syllabus list):
+
+- `1ANN_MNIST.ipynb` - ANN on MNIST, worked through in class
+- `Image_basics.ipynb` - reading and handling images with Matplotlib, OpenCV and Pillow (uses `Night_City.png`)
+
+These two need a few extra libraries and the MNIST arrays as `.npy` files, see [Class notebooks](#class-notebooks) below.
+
+`datasets/ecg.csv` and `datasets/cbow_corpus.txt` are already in the repo. MNIST, CIFAR-10 and the MobileNetV2 weights are **not** (they are large), `prepare_offline.py` downloads them for you.
+
+## Before you start (Windows)
+
+You need:
+
+1. **Windows 10/11, 64-bit**, about 4 GB free disk space, 8 GB RAM or more
+2. **Python 3.12 (64-bit)** from https://www.python.org/downloads/windows/
+   - tick **"Add python.exe to PATH"** on the first screen of the installer
+   - do not use 3.13 / 3.14, TensorFlow 2.19 does not support them yet
+   - do not use the Microsoft Store version
+3. **Microsoft Visual C++ Redistributable (x64)**: https://aka.ms/vs/17/release/vc_redist.x64.exe (TensorFlow needs it)
+4. **Git for Windows**: https://git-scm.com/download/win (keep the default options while installing)
+5. A **GitHub account**: https://github.com/signup
+
+Check in Command Prompt:
+
+```
+py -3.12 --version
+git --version
+```
+
+## Step 1: Fork the repo (make your own copy on GitHub)
+
+A fork is your own copy of this repo under your GitHub account. You can change anything in your fork and push your work there, without touching the original.
+
+1. Log in to GitHub.
+2. Open https://github.com/AdityaThakare72/DVVPCOE_DL_SEP2026
+3. Click **Fork** (top right) -> keep the name as it is -> click **Create fork**.
+4. You are now on `https://github.com/<your-username>/DVVPCOE_DL_SEP2026`. This is your copy.
+
+## Step 2: Clone your fork to your laptop
+
+Clone **your fork**, not the original repo, otherwise you will not be able to push your work.
+
+Open **Command Prompt** (Windows key -> type `cmd` -> Enter) and run, replacing `<your-username>`:
+
+```
+mkdir C:\DL
+cd /d C:\DL
+git clone https://github.com/<your-username>/DVVPCOE_DL_SEP2026.git
+cd DVVPCOE_DL_SEP2026
+```
+
+Keep the path short like `C:\DL\...`. Avoid Desktop / Documents / OneDrive folders, OneDrive syncing locks files during installation and long paths break TensorFlow.
+
+Tell Git who you are (one time only, use your GitHub name and email):
+
+```
+git config --global user.name "Your Name"
+git config --global user.email "you@example.com"
+```
+
+## Step 3: Install everything and download the data
+
+Needs internet, about 800 MB download, 20-40 minutes. Do this at home, before the session.
+
+In Command Prompt inside the repo folder:
+
+```
+py -3.12 -m venv dl_env
+dl_env\Scripts\activate
+python -m pip install --upgrade pip
+python -m pip install -r requirements.txt
+python prepare_offline.py
+python check_setup.py
+```
+
+Every line of `check_setup.py` must start with `[PASS]` and the last line must be `READY`.
+
+Then test it offline: turn off Wi-Fi and run `python check_setup.py` again. It should still say `READY`.
+
+## Step 4: Run the notebooks
+
+Every time:
+
+```
+cd /d C:\DL\DVVPCOE_DL_SEP2026
+dl_env\Scripts\activate
+jupyter notebook
+```
+
+A browser tab opens. Open a notebook and run cells one by one with **Shift + Enter**. Keep the black Command Prompt window open, closing it stops Jupyter.
+
+Always start Jupyter from the repo folder. If you start it from somewhere else, the notebooks stop with `datasets folder not found`.
+
+Approx. training time on a normal laptop CPU:
+
+| Practical | Time | If short on time, reduce |
+|---|---|---|
+| 2 | 1-2 min | `EPOCHS` |
+| 3 | 5-12 min | `EPOCHS` |
+| 4 | < 1 min | - |
+| 5 | < 1 min | - |
+| 6 | 6-12 min | `N_TRAIN`, `HEAD_EPOCHS`, `FINE_EPOCHS` |
+
+## Step 5: Save your work to your fork
+
+After you run / change notebooks, push them to your fork:
+
+```
+git add .
+git commit -m "practical 2 done"
+git push
+```
+
+The first push opens a browser window asking you to log in to GitHub. Log in and allow it.
+
+The downloaded datasets, the `models/` folder and `dl_env/` are in `.gitignore`, so they will not be uploaded (they are hundreds of MB). Only your notebooks and code go to GitHub. Do not remove those lines from `.gitignore`.
+
+## Getting updates from the original repo
+
+If new material is added to this repo later, bring it into your fork.
+
+**Easy way (on the website):** open your fork on GitHub -> click **Sync fork** -> **Update branch**. Then on your laptop:
+
+```
+git pull
+```
+
+**Command line way:**
+
+```
+git remote add upstream https://github.com/AdityaThakare72/DVVPCOE_DL_SEP2026.git
+git fetch upstream
+git merge upstream/master
+git push
+```
+
+(`git remote add upstream ...` is needed only the first time.)
+
+If you changed the same notebook that was updated in the original repo, Git may report a merge conflict. The simplest fix is to copy your version of the notebook somewhere safe, take the original version, and redo your changes on top of it.
+
+## Class notebooks
+
+`1ANN_MNIST.ipynb` and `Image_basics.ipynb` need a few libraries that are not in `requirements.txt`. With `dl_env` active:
+
+```
+python -m pip install seaborn opencv-python scikit-image
+```
+
+`1ANN_MNIST.ipynb` reads MNIST from `datasets/mnist/*.npy`. After `prepare_offline.py` has downloaded `datasets/mnist.npz`, create those files once:
+
+```
+python -c "import numpy as np, os; d=np.load('datasets/mnist.npz'); os.makedirs('datasets/mnist', exist_ok=True); [np.save(f'datasets/mnist/{k}.npy', d[k]) for k in d.files]"
+```
+
+## Common problems
+
+| Problem | Fix |
+|---|---|
+| `py` is not recognized | Reinstall Python 3.12 from python.org with "Add python.exe to PATH" ticked, then open a new Command Prompt |
+| `No matching distribution found for tensorflow==2.19.1` | Wrong Python version. Delete `dl_env` folder, recreate with `py -3.12 -m venv dl_env` |
+| `DLL load failed while importing _pywrap_tensorflow_internal` | Install the Visual C++ Redistributable, restart the laptop |
+| PowerShell says running scripts is disabled | Use Command Prompt (cmd) instead of PowerShell |
+| `No module named tensorflow` inside a notebook | Jupyter was started without `dl_env` active. Close it, run `dl_env\Scripts\activate`, then `jupyter notebook` |
+| `datasets folder not found` | Jupyter was started from another folder. Start it from the repo folder |
+| `[FAIL]` on a dataset line | Run `python prepare_offline.py` again with internet |
+| `git push` gives permission denied (403) | You cloned the original repo instead of your fork. Check with `git remote -v`, it must show your username |
+| Training is very slow | Close other programs, reduce `EPOCHS` / `N_TRAIN` |
+
+## Syllabus
+
+SPPU Final Year Information Technology (2019 Course), 414447: Lab Practice IV, Group A (Deep Learning), assignments 1 to 6.
+
+Reference books (as per syllabus):
+- Hands-On Deep Learning Algorithms with Python - Sudharsan Ravichandiran
+- Python Deep Learning, 2nd Edition - Ivan Vasilev et al.
+- Natural Language Processing with Python Quick Start Guide - Mirant Kasliwal
+
+---
+
+Aditya Thakare - https://github.com/AdityaThakare72
-- 
2.43.0