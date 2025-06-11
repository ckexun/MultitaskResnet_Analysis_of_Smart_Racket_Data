# MultitaskResNet: Smart Racket Data Analysis
>Update time: 2025-06-11

---
This project implements **MultitaskResNet**, a deep learning model based on multi-task learning to analyze swing motion data collected from smart rackets. It was developed for the **Contest** and achieved excellent results on both the public and private leaderboards.



## Document

### Setup Environment
![Python Badge](https://img.shields.io/badge/Python-3.10.16-blue)

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python main.py
```

### Visualize Training (TensorBoard)



## Requirements

You can install dependencies in two ways:

### Option 1: Using pip
```bash
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda env create -f environment.yml
conda activate tennis
```

The main packages include:
- PyTorch with CUDA 12.4
- NumPy, pandas, scikit-learn
- TensorBoard
- tqdm...  

- Python = 3.10.16


You can create a `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

## Dataset

> ⚠️ **Note**: The dataset files (`39_Training_Dataset` and `39_Test_Dataset`) are too large to be stored in this repository.
>
> Please download them manually from the official competition page:  
> 👉 https://tbrain.trendmicro.com.tw/Competitions/Details/39 -> Download Dataset -> Download 39_Training_Dataset and 39_Test_Dataset

## Model Overview

The model predicts:
- Gender (binary)
- Hold racket handed (binary)
- Years of experience (3-class)
- Level (4-class)

Losses include **Focal Loss** (for binary tasks) and **Label Smoothing CrossEntropy** (for multi-class).



## Experiments

Check the `experiments/` folder for traditional ML model implementations (e.g., Random Forest, CatBoost). Results and submission files are stored in `experiments/*/result/`.


## Project Structure

```
MultitaskResnet_Analysis_of_Smart_Racket_Data/
├── main.py              # Main model and training/inference pipeline
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

## Notes

- The model includes CNN blocks, residual connections, and self-attention.
- Data is normalized and optionally augmented with Gaussian noise and time reversal.