
import numpy as np
import pandas as pd
import os, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from decimal import Decimal, ROUND_HALF_UP  # Recommended to use this method

# ----------------------- Hyperparameters -----------------------
TRAIN_DATA_DIR = "./dataset/39_Training_Dataset/train_data"
TRAIN_INFO_PATH = "./dataset/39_Training_Dataset/train_info.csv"
TEST_DATA_DIR = "./dataset/39_Test_Dataset/test_data"
TEST_INFO_PATH = "./dataset/39_Test_Dataset/test_info.csv"
SUBMISSION_FILE = "submission_improved_staged.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 2500
BATCH_SIZE = 16

# ---- Stage 1: Super-Augmented ----
STAGE1_EPOCHS = 40
STAGE1_LR = 1e-4
STAGE1_PATIENCE = 12
STAGE1_DROPOUT = 0.5
STAGE1_MIXUP_ALPHA = 0.3
STAGE1_LABEL_SMOOTHING = 0.07
STAGE1_NOISE_STD = 0.04
STAGE1_TIME_REVERSE_PROB = 0.3

# ---- Stage 2: Fine-tuning ----
STAGE2_EPOCHS = 40
STAGE2_LR = 4e-5
STAGE2_PATIENCE = 8
STAGE2_DROPOUT = 0.3
STAGE2_MIXUP_ALPHA = 0.05
STAGE2_LABEL_SMOOTHING = 0.02
STAGE2_NOISE_STD = 0.01
STAGE2_TIME_REVERSE_PROB = 0.1

# ---------------------------------------------------------------

# ----------- Dataset + Augmentation -----------
class SwingDataset(Dataset):
    def __init__(self, X, y_gender, y_hand, y_years, y_level,
                 augment=False, noise_std=0.01, time_reverse_prob=0.1):
        self.X = X
        self.y_gender = y_gender
        self.y_hand = y_hand
        self.y_years = y_years
        self.y_level = y_level
        self.augment = augment
        self.noise_std = noise_std
        self.time_reverse_prob = time_reverse_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # Add Gaussian noise with probability 0.5
            if random.random() < 0.5:
                x = x + np.random.normal(0, self.noise_std, x.shape)
            # Time reverse with specified probability
            if random.random() < self.time_reverse_prob:
                x = x[::-1, :]
        x = x.copy()  # Prevent negative stride errors
        return torch.tensor(x, dtype=torch.float32), {
            'gender': torch.tensor(self.y_gender[idx], dtype=torch.float32),
            'hand': torch.tensor(self.y_hand[idx], dtype=torch.float32),
            'years': torch.tensor(self.y_years[idx], dtype=torch.long),
            'level': torch.tensor(self.y_level[idx], dtype=torch.long)
        }

# ------------ Mixup -----------
def mixup_data(x, y_dict, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_mixed = {}
    for task, y in y_dict.items():
        if task in ['gender', 'hand']:
            y_mixed[task] = lam * y + (1 - lam) * y[index]
        else:
            y_mixed[task] = (y, y[index], lam)
    return mixed_x, y_mixed

# --------- Losses -----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()

# ------------- Model -------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Conv1d(embed_size, embed_size, kernel_size=1)
        self.key = nn.Conv1d(embed_size, embed_size, kernel_size=1)
        self.value = nn.Conv1d(embed_size, embed_size, kernel_size=1)
        self.scale = torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))

    def forward(self, x):
        batch_size, embed_size, seq_len = x.size()
        Q = self.query(x).permute(0,2,1)
        K = self.key(x).permute(0,2,1)
        V = self.value(x).permute(0,2,1)
        K_T = K.transpose(1,2)
        attn = torch.bmm(Q, K_T) / self.scale.to(x.device)
        attn = F.softmax(attn, dim=2)
        out = torch.bmm(attn, V).permute(0,2,1)
        return out

class MultitaskResNet(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block1 = ResidualBlock(64, 128, stride=2)
        self.res_block2 = ResidualBlock(128, 256, stride=2)
        self.attention1 = SelfAttention(256)
        self.res_block3 = ResidualBlock(256, 512, stride=2)
        self.attention2 = SelfAttention(512)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.gender_features = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate))
        self.gender_head = nn.Linear(256, 1)
        self.hand_features = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate))
        self.hand_head = nn.Linear(256, 1)
        self.years_features = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate))
        self.years_head = nn.Linear(256, 3) # years: 0, 1, 2
        self.level_features = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate))
        self.level_head = nn.Linear(256, 4) # level: 2, 3, 4, 5

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.attention1(x)
        x = self.res_block3(x)
        x = self.attention2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        gender_feat = self.gender_features(x)
        gender_out = self.gender_head(gender_feat).squeeze(-1)
        hand_feat = self.hand_features(x)
        hand_out = self.hand_head(hand_feat).squeeze(-1)
        years_feat = self.years_features(x)
        years_out = self.years_head(years_feat)
        level_feat = self.level_features(x)
        level_out = self.level_head(level_feat)
        return {
            'gender': gender_out,
            'hand': hand_out,
            'years': years_out,
            'level': level_out
        }

# --------- Training/Eval -----------
def train_epoch(model, dataloader, optimizer, device, mixup_alpha, label_smoothing):
    model.train()
    criterion_gender = FocalLoss(alpha=0.25, gamma=2)
    criterion_hand = FocalLoss(alpha=0.25, gamma=2)
    criterion_years = LabelSmoothingCE(smoothing=label_smoothing)
    criterion_level = LabelSmoothingCE(smoothing=label_smoothing)
    total_loss = 0
    for x, y in tqdm(dataloader, leave=False):
        x = x.to(device)
        y_gender = y['gender'].to(device)
        y_hand = y['hand'].to(device)
        y_years = y['years'].to(device)
        y_level = y['level'].to(device)
        y_dict = {'gender': y_gender, 'hand': y_hand, 'years': y_years, 'level': y_level}

        if mixup_alpha > 0:
            x, mixed_y = mixup_data(x, y_dict, mixup_alpha)
            y_gender = mixed_y['gender']
            y_hand = mixed_y['hand']
            y_years_a, y_years_b, lam_years = mixed_y['years']
            y_level_a, y_level_b, lam_level = mixed_y['level']

        optimizer.zero_grad()
        out = model(x)

        loss_gender = criterion_gender(out['gender'], y_gender)
        loss_hand = criterion_hand(out['hand'], y_hand)
        if mixup_alpha > 0:
            loss_years = lam_years * criterion_years(out['years'], y_years_a) + (1 - lam_years) * criterion_years(out['years'], y_years_b)
            loss_level = lam_level * criterion_level(out['level'], y_level_a) + (1 - lam_level) * criterion_level(out['level'], y_level_b)
        else:
            loss_years = criterion_years(out['years'], y_years)
            loss_level = criterion_level(out['level'], y_level)

        loss = loss_gender + loss_hand + loss_years + loss_level
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, label_smoothing):
    model.eval()
    y_true = {'gender': [], 'hand': [], 'years': [], 'level': []}
    y_pred = {'gender': [], 'hand': [], 'years': [], 'level': []}
    criterion_gender = FocalLoss(alpha=0.25, gamma=2)
    criterion_hand = FocalLoss(alpha=0.25, gamma=2)
    criterion_years = LabelSmoothingCE(smoothing=label_smoothing)
    criterion_level = LabelSmoothingCE(smoothing=label_smoothing)
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y_gender = y['gender'].to(device)
            y_hand = y['hand'].to(device)
            y_years = y['years'].to(device)
            y_level = y['level'].to(device)
            out = model(x)
            y_true['gender'].extend(y_gender.cpu().numpy())
            y_true['hand'].extend(y_hand.cpu().numpy())
            y_true['years'].extend(y_years.cpu().numpy())
            y_true['level'].extend(y_level.cpu().numpy())
            y_pred['gender'].extend(torch.sigmoid(out['gender']).cpu().numpy())
            y_pred['hand'].extend(torch.sigmoid(out['hand']).cpu().numpy())
            y_pred['years'].extend(torch.softmax(out['years'], dim=1).cpu().numpy())
            y_pred['level'].extend(torch.softmax(out['level'], dim=1).cpu().numpy())
            loss_gender = criterion_gender(out['gender'], y_gender)
            loss_hand = criterion_hand(out['hand'], y_hand)
            loss_years = criterion_years(out['years'], y_years)
            loss_level = criterion_level(out['level'], y_level)
            total_loss += (loss_gender + loss_hand + loss_years + loss_level).item()
    auc_gender = roc_auc_score(y_true['gender'], y_pred['gender'])
    auc_hand = roc_auc_score(y_true['hand'], y_pred['hand'])
    auc_years = roc_auc_score(y_true['years'], y_pred['years'], multi_class='ovr')
    auc_level = roc_auc_score(y_true['level'], y_pred['level'], multi_class='ovr')
    avg_loss = total_loss / len(dataloader)
    print(f"AUC Gender: {auc_gender:.4f}, Hand: {auc_hand:.4f}, Years: {auc_years:.4f}, Level: {auc_level:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train_staged(model, train_loader, val_loader, optimizer, device, num_epochs, patience,
                 mixup_alpha, label_smoothing, stage_name):
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop_counter = 0
    best_loss = float('inf')
    best_epoch = -1
    writer = SummaryWriter(log_dir=f"runs/{stage_name}")

    for epoch in range(num_epochs):
        print(f"\n[{stage_name}] Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, device, mixup_alpha, label_smoothing
        )
        val_loss = evaluate(model, val_loader, device, label_smoothing)
        scheduler.step()
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'best_model_{stage_name}.pth')
            print(f"Best model updated at epoch {best_epoch} with val_loss = {best_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    writer.close()
    print(f"Training completed. Best model was from epoch {best_epoch} with val_loss = {best_loss:.4f}")
    return best_loss, best_epoch

# ---------- Inference ----------
def inference(model, dataloader, device, submission_file="submission_improved_staged.csv"):
    # predict
    model.eval()
    preds = []
    with torch.no_grad():
        for x in tqdm(dataloader):
            x = x.to(device)
            out = model(x)
            gender = (1 - torch.sigmoid(out['gender'])).cpu().numpy()
            hand = (1 - torch.sigmoid(out['hand'])).cpu().numpy()
            years = torch.softmax(out['years'], dim=1).cpu().numpy()
            level = torch.softmax(out['level'], dim=1).cpu().numpy()
            for i in range(x.shape[0]):
                preds.append(
                    [gender[i], hand[i]] + years[i].tolist() + level[i].tolist()
                )

    # Output in the competition-specified file format
    columns = [
        'gender', 
        'hold racket handed',
        'play years_0', 'play years_1', 'play years_2',
        'level_2', 'level_3', 'level_4', 'level_5'
    ]
    float_format = lambda x: str(Decimal(x).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
    
    df = pd.DataFrame(preds, columns=columns)
    df.insert(0, 'unique_id', unique_ids)
    # df.iloc[:, 1:] = df.iloc[:, 1:].astype(object).applymap(float_format)
    # df.to_csv(submission_file, index=False, encoding="utf-8", lineterminator='\n')
    for col in df.columns[1:]:
        df[col] = df[col].apply(float_format)
    with open(submission_file, "w", encoding="utf-8", newline='\n') as f:
        df.to_csv(f, index=False)
    print(f"Successful! Submission saved to {submission_file}")
    print("Done!")

# ------------- Utility Functions --------------
def load_data(file_path, seq_len=SEQ_LEN, random_crop=True):
    data = np.loadtxt(file_path)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    T = data.shape[0]
    if T >= seq_len:
        start = np.random.randint(0, T - seq_len + 1) if random_crop else 0  
        data = data[start:start + seq_len]
    else:
        pad_width = seq_len - T
        data = np.pad(data, ((0, pad_width), (0, 0)), mode='constant')
    return data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # set_seed(42)
    
    print("Loading training data...")
    train_info_df = pd.read_csv(TRAIN_INFO_PATH)
    """
    unique_id：選手揮拍測驗的id(對應train_data檔名)
    player_id：選手id
    mode：揮拍模式(參照「程式介紹.ppt」第六頁說明)
    gender : 1:男,2:女
    hold racket handed : 1:右 2:左
    play years : 共3個球齡層(根據所有選手的球齡分布，分為 0:低、1:中、2:高)
    level : 共4個等級(2:大專甲組選手、3:大專乙組選手、4:青少年國手、5:青少年選手)
    cut_point：每次揮拍的資料切割節點，僅供參考
    """

    # All are integer types, just re-encode the columns directly
    train_info_df['gender'] -= 1
    train_info_df['hand'] = train_info_df['hold racket handed'] - 1
    # train_info_df['play years'] = train_info_df['play years'] - 0
    train_info_df['level'] -= 2
    X_swing_data = []
    y_gender, y_hand, y_years, y_level = [], [], [], []
    for _, row in tqdm(train_info_df.iterrows(), total=len(train_info_df)):
        try:
            unique_id = row['unique_id']
            file_path = os.path.join(TRAIN_DATA_DIR, f"{unique_id}.txt")
            x = load_data(file_path)
            X_swing_data.append(x)
            y_gender.append(row['gender'])
            y_hand.append(row['hand'])
            y_years.append(row['play years'])
            y_level.append(row['level'])
        except Exception as e:
            print(f"Error processing {unique_id}: {e}")
            continue
    X_swing_data = np.array(X_swing_data)
    y_gender = np.array(y_gender)
    y_hand = np.array(y_hand)
    y_years = np.array(y_years)
    y_level = np.array(y_level)

    print("Data loading completed, splitting into training and validation sets...")
    X_train, X_val, y_gender_train, y_gender_val, y_hand_train, y_hand_val, y_years_train, y_years_val, y_level_train, y_level_val = train_test_split(
        X_swing_data, y_gender, y_hand, y_years, y_level, test_size=0.2, random_state=42
    )

    # ---- Stage 1: Super Augmentation ----
    print("\n==== Stage 1: Super Augmented Training ====")
    dataset_train1 = SwingDataset(X_train, y_gender_train, y_hand_train, y_years_train, y_level_train,
                                        augment=True, noise_std=STAGE1_NOISE_STD, time_reverse_prob=STAGE1_TIME_REVERSE_PROB)
    dataset_val = SwingDataset(X_val, y_gender_val, y_hand_val, y_years_val, y_level_val,
                                     augment=False, noise_std=0.0, time_reverse_prob=0.0)
    dataloader_train1 = DataLoader(dataset_train1, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    model = MultitaskResNet(dropout_rate=STAGE1_DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=1e-4)
    best_loss1, best_epoch1 = train_staged(
        model, dataloader_train1, dataloader_val, optimizer, device, STAGE1_EPOCHS, STAGE1_PATIENCE,
        mixup_alpha=STAGE1_MIXUP_ALPHA, label_smoothing=STAGE1_LABEL_SMOOTHING, stage_name="stage1"
    )

    # ---- Stage 2: Fine-tuning ----
    print("\n==== Stage 2: Fine-tuning on Less Noisy Data ====")
    dataset_train2 = SwingDataset(X_train, y_gender_train, y_hand_train, y_years_train, y_level_train,
                                        augment=True, noise_std=STAGE2_NOISE_STD, time_reverse_prob=STAGE2_TIME_REVERSE_PROB)
    dataloader_train2 = DataLoader(dataset_train2, batch_size=BATCH_SIZE, shuffle=True)
    # Re-create model and optimizer for fine-tuning
    model = MultitaskResNet(dropout_rate=STAGE2_DROPOUT).to(device)
    model.load_state_dict(torch.load("best_model_stage1.pth"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=1e-4)
    best_loss2, best_epoch2 = train_staged(
        model, dataloader_train2, dataloader_val, optimizer, device, STAGE2_EPOCHS, STAGE2_PATIENCE,
        mixup_alpha=STAGE2_MIXUP_ALPHA, label_smoothing=STAGE2_LABEL_SMOOTHING, stage_name="stage2"
    )
    print(f"\nTraining completed! The best model in Stage 2 was at epoch {best_epoch2}, with validation loss = {best_loss2:.4f}")


    # ---- Test Prediction & Submission ----
    print("\nPreparing to perform test set prediction...")
    unique_ids, X_test_data = [], []
    test_info = pd.read_csv(TEST_INFO_PATH)
    for _, row in tqdm(test_info.iterrows(), total=len(test_info)):
        unique_id = row['unique_id']
        file_path = os.path.join(TEST_DATA_DIR, f"{unique_id}.txt")
        try:
            # x = load_data(file_path)                   # best model use this
            x = load_data(file_path, random_crop=False)  # Avoid inference inconsistency due to random (when seed is set)
            X_test_data.append(torch.tensor(x, dtype=torch.float32))
            unique_ids.append(unique_id)
        except Exception as e:
            print(f"Error processing {unique_id}: {e}")
            continue
    print(f"Loaded {len(unique_ids)} test samples")
    model = MultitaskResNet(dropout_rate=STAGE2_DROPOUT).to(device)
    model.load_state_dict(torch.load("best_model_stage2.pth"))
    X_test_data = torch.stack(X_test_data)
    dataloader = DataLoader(X_test_data, batch_size=32, shuffle=False)
    inference(model, dataloader, device, SUBMISSION_FILE)
    
    # Can visualize the results in TensorBoard using the data in the 'runs' directory.
    # The training process loss (train/val) will be saved in the runs/ directory.
    # Can launch TensorBoard for visualization by running: tensorboard --logdir=runs
