LP-IV DEEP LEARNING PRACTICALS (SPPU BE IT 2019 pattern, 414447 Lab Practice IV)
===============================================================================
Target lab setup: Windows, Python 3.12, Jupyter Notebook, NO internet during the session.
Libraries: TensorFlow 2.19.1 + Keras 3 only.

FOLDER CONTENTS
---------------
Practical_2_Feedforward_NN_MNIST.ipynb               Feedforward NN, SGD, MNIST (CIFAR-10 switch inside)
Practical_3_Image_Classification_CNN_CIFAR10.ipynb   CNN in 4 stages, CIFAR-10
Practical_4_Autoencoder_Anomaly_Detection_ECG.ipynb  Autoencoder anomaly detection, ECG5000
Practical_5_CBOW_Word_Embeddings.ipynb               CBOW word embeddings, included text corpus
Practical_6_Transfer_Learning_MobileNetV2.ipynb      Transfer learning + fine-tuning, MobileNetV2, CIFAR-10
requirements.txt        Exact library versions
prepare_offline.py      Downloads all datasets and weights (run ONCE with internet)
check_setup.py          Verifies a lab machine is ready (run offline)
datasets/               Data files (filled by prepare_offline.py; cbow_corpus.txt already included)
models/                 Pretrained MobileNetV2 weights (filled by prepare_offline.py)
(Practical 1 is covered by the separate presentation.)

DATASETS AND FILES NEEDED
-------------------------
File                                   Used in     Size     Source
datasets/mnist.npz                     P2          11 MB    https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
datasets/cifar10.npz                   P2*, P3, P6 ~150 MB  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  (converted by the script)
datasets/ecg.csv                       P4          5 MB     http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
datasets/cbow_corpus.txt               P5          6 KB     included, no download
models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
                                       P6          9 MB     https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
* P2 uses CIFAR-10 only if DATASET = "cifar10" is chosen.

The notebooks read ONLY from these local folders. Nothing is downloaded while they run.

ONE-TIME PREPARATION (on any machine WITH internet)
---------------------------------------------------
1. Install Python 3.12 (64-bit) from python.org. Tick "Add python.exe to PATH".
2. Open Command Prompt inside this folder and run:
       pip install -r requirements.txt
       python prepare_offline.py
   This fills datasets/ and models/.
3. Download the library installers (wheels) for offline lab machines.
   Do this on a WINDOWS machine with Python 3.12 so the correct files are fetched:
       pip download -r requirements.txt -d wheels
   This creates a wheels/ folder (several hundred MB; TensorFlow alone is the largest file).

LAB MACHINE SETUP (no internet)
-------------------------------
1. Copy the whole folder (including datasets/, models/, wheels/) to the lab machine.
2. Install Python 3.12 (64-bit) from the saved installer.
3. In Command Prompt inside the folder:
       pip install --no-index --find-links wheels -r requirements.txt
4. Verify:
       python check_setup.py
   Every line must say PASS and the last line must say READY.
5. Start Jupyter from the same folder (important: notebooks look for datasets/ next to them):
       jupyter notebook

Optional: use a virtual environment so the lab's other Python work is not affected:
       python -m venv dl_env
       dl_env\Scripts\activate
   then run the pip install command above.

APPROXIMATE RUN TIMES ON A LAB CPU (4 cores)
--------------------------------------------
P2: 1-2 min   P3: 5-12 min   P4: under 1 min   P5: under 1 min   P6: 6-12 min
Reduce EPOCHS (P2, P3) or N_TRAIN / HEAD_EPOCHS / FINE_EPOCHS (P6) if time is short.

COMMON PROBLEMS
---------------
- "datasets folder not found": Jupyter was started from another folder. Close it and start it
  from this folder, or move the notebook next to datasets/.
- "DLL load failed" when importing tensorflow: install the Microsoft Visual C++ Redistributable
  (x64) on the lab machine (download the installer in advance: https://aka.ms/vs/17/release/vc_redist.x64.exe).
- "Could not find a version that satisfies the requirement" during offline install: the wheels
  were downloaded on a machine with a different Python version or OS. Download again on
  Windows + Python 3.12.
- Very slow training: close other programs, or reduce the EPOCHS / N_TRAIN values.
