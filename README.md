# MultitaskResNet: Smart Racket Data Analysis
>Update time: 2025-08-02

This project implements **MultitaskResNet**, a deep learning model based on multi-task learning to analyze swing motion data collected from smart rackets. It was developed for the **Contest** and achieved excellent results on both the public and private leaderboards.

---

## Document

### Setup Environment
![Python Badge](https://img.shields.io/badge/Python-3.10.16-blue)

You can install dependencies in two ways:

#### Option 1: Using pip (venv)

This project does not specify a particular CUDA version in `requirements.txt`. Please install PyTorch manually according to your hardware environment.

```bash
pip install -r requirements.txt
```
📌 After installing the basic dependencies, please install PyTorch manually from https://pytorch.org/ to match your system's CUDA or CPU configuration.


#### Option 2: Using conda 

```bash
conda env create -f environment.yml
conda activate tennis
```

> ⚠️ This project uses **PyTorch with CUDA 12.4**  
> 💡 Please ensure you have the appropriate **CUDA drivers** installed to match the CUDA runtime version.

### Train and Inference the Model

```bash
python main.py
```

### Visualize Training (TensorBoard)


---

## Requirements
Main packages:
- Python 3.10.16
- PyTorch (with CUDA 12.4 — make sure compatible driver is installed)
- pandas, numpy, scikit-learn, tqdm
- tensorboard

Install via pip or conda (see Setup section).

---

## Project Structure

```
MultitaskResnet_Analysis_of_Smart_Racket_Data/
├── main.py                    # Main model and training/inference pipeline
├── dataset/                   # Contains training and test datasets
│   ├── 39_Training_Dataset/
│   ├── 39_Test_Dataset/
│   ├── Readme(train).txt
│   └── Readme(test).txt
├── baseline/                  # Legacy or baseline code implementations
│   └── old/
├── experiments/               # Experiments with other models like Random Forest, CatBoost
├── requirements.txt           # Pip-based package dependencies
├── environment.yml            # Conda environment definition (name: tennis)
└── README.md                  # Project documentation

```

---

## Dataset

> ⚠️ **Note**: The dataset files (`39_Training_Dataset` and `39_Test_Dataset`) are too large to be stored in this repository.

Please download them manually from the official competition page:  
👉 https://tbrain.trendmicro.com.tw/Competitions/Details/39  
→ **Download Dataset** → Download `39_Training_Dataset` and `39_Test_Dataset`

After downloading, place them in the `dataset/` directory as follows:

```
dataset/
├── 39_Training_Dataset/
│   ├── train_info.csv
│   └── train_data/*.txt
├── 39_Test_Dataset/
│   ├── test_info.csv
│   └── test_data/*.txt
```

---

## How to Run

To train and generate submission in one step:

```bash
python main.py
```

This will:
- Load and normalize the training data
- Stage 1: Train with strong data augmentation
- Stage 2: Fine-tune with weaker noise
- Inference
- Generate `submission_improved_staged.csv`

---

##  Customize Hyperparameters

You can modify the training strategy by editing the hyperparameters at the top of [`main.py`](./main.py).  
Example fields include:

```python
SEQ_LEN = 2500
BATCH_SIZE = 16

# Stage 1: Strong augmentation
STAGE1_EPOCHS = 40
STAGE1_LR = 1e-4
STAGE1_DROPOUT = 0.5
STAGE1_MIXUP_ALPHA = 0.3

# Stage 2: Fine-tuning
STAGE2_EPOCHS = 40
STAGE2_LR = 4e-5
STAGE2_DROPOUT = 0.3
```

To retrain the model with updated hyperparameters:

```bash
python main.py
```

This will train from scratch and overwrite previous checkpoints if any.

---

## Code Structure

The `main.py` script includes:
- Data loading and preprocessing
- Stage 1 training (strong noise + mixup)
- Stage 2 fine-tuning (low noise)
- Multi-task prediction (gender, hand, years, level)
- TensorBoard logging and checkpoint saving

---

## Model Overview

The model predicts the following targets:
- Gender (binary)
- Hold racket handed (binary)
- Years of experience (3-class)
- Level (4-class)

Model architecture:
- Residual Blocks
- Self-Attention Layers
- Focal Loss (binary)
- Label Smoothing CrossEntropy (multi-class)

---

## Output Format

After inference, a submission CSV is generated as `submission_improved_staged.csv`:

```csv
unique_id,gender,hold racket handed,play years_0,play years_1,play years_2,level_2,level_3,level_4,level_5
1968,0.9293,0.9333,...,...,...,...,...,...,...
```

---

## Experiments

Check the `experiments/` folder for traditional ML model implementations (e.g., Random Forest, CatBoost). Results and submission files are stored in `experiments/*/result/`.

---

## Notes

- The model includes CNN blocks, residual connections, and self-attention.
- Data is normalized and optionally augmented with Gaussian noise and time reversal.
