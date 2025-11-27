ACTION RECOGNITION FOR VIDEO SURVEILLANCE – PROJECT
===================================================

This project contains the implementation of the action-recognition system
described in the report submitted for the Computer Vision course.
The system uses a CNN+LSTM architecture to classify human actions
from short video clips in a surveillance-style environment.

The repository includes:
- Full training and evaluation pipeline
- Demo script producing an annotated output video
- Model architecture (CNN + LSTM)
- Preprocessing utilities
- Requirements file


---------------------------------------------------
PROJECT STRUCTURE
---------------------------------------------------

action_recognition_project/
    train_eval.py          → Training + evaluation pipeline
    demo.py                → Script to generate demo video
    models/
        cnn_lstm.py        → CNN + LSTM model
    utils/
        preprocess.py      → Frame extraction & transforms
    requirements.txt       → Python package requirements
    README.txt             → This file


---------------------------------------------------
1. HOW TO REPRODUCE THE RESULTS
---------------------------------------------------

The following instructions describe exactly how to recreate
the results reported in the project document, including the
training metrics and the validation tables.


---------------------------------------------------
(1) CREATE AND ACTIVATE A VIRTUAL ENVIRONMENT
---------------------------------------------------

python -m venv .venv

Windows:
    .\.venv\Scripts\activate

Linux/Mac:
    source .venv/bin/activate


---------------------------------------------------
(2) INSTALL THE DEPENDENCIES
---------------------------------------------------

pip install --upgrade pip
pip install -r requirements.txt

This installs:
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- TQDM
and all utilities required by the pipeline.


---------------------------------------------------
(3) PREPARE THE DATASET
---------------------------------------------------

The dataset must follow a folder structure such as:

DATA_ROOT/
    class_1/
        video1.mp4
        video2.mp4
    class_2/
        video3.mp4

Each subfolder corresponds to one action class.
The list of classes used by the code is defined in:

    train_eval.py  →   classes = [...]

DATA_ROOT is passed as an argument.


---------------------------------------------------
(4) RUN TRAINING + EVALUATION
---------------------------------------------------

python train_eval.py --data_root DATA_ROOT --epochs N

This script:
- Loads and preprocesses videos
- Extracts frame sequences
- Trains the CNN+LSTM model
- Validates on a held-out split
- Computes:
    • Training loss
    • Validation loss
    • Training accuracy
    • Validation accuracy
    • Macro F1 (train/val)
    • Confusion matrix

All metrics are automatically saved to:

    results_metrics.csv

The best model is stored at:

    saved_models/cnn_lstm_best.pth


---------------------------------------------------
(5) GENERATE THE DEMO VIDEO
---------------------------------------------------

python demo.py

This script loads the trained model and processes a sample video.
It overlays predicted action labels onto the frames and outputs:

    output_demo.mp4

This is the video included in the submission.


---------------------------------------------------
DEMO VIDEO
---------------------------------------------------

Link to demonstration video:

https://drive.google.com/file/d/16yCIIFquuSWKGYACpiAHIH3KfgoE_VRK/view?usp=sharing

---------------------------------------------------
REPOSITORY
---------------------------------------------------

GitHub project link:

https://github.com/JoseLorenzo97/action_recognition_project