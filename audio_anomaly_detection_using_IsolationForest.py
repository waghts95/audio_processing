# ===============================================================
# 🧠 Simple Audio Anomaly Detection (IsolationForest)
# Dataset: DCASE2023 Task 2 - FAN Machine
# Source: https://zenodo.org/records/7882613
# ---------------------------------------------------------------
# Folder structure:
# fan/
# ├── train/   (only normal sounds)
# └── test/    (normal + abnormal sounds)
# ===============================================================

import os, random, librosa, numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1️⃣ CONFIGURATION
# -----------------------------
BASE_DIR = "/home/tushar/work/datasets/audio/fan"  # change to your path
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

N_TRAIN = 10   # number of normal files to use for training
N_TEST  = 20   # number of test files (normal + abnormal)

# -----------------------------
# 2️⃣ FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path):
    """Extract MFCC + basic spectral features from a WAV file"""
    try:
        y, sr = librosa.load(file_path, sr=None, duration=3, offset=0.5)
        if y.size == 0:
            return None
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        return np.hstack([mfcc, centroid, bandwidth, contrast, rolloff, zcr])
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# -----------------------------
# 3️⃣ LOAD TRAINING DATA (NORMAL)
# -----------------------------
train_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(".wav")]
random.shuffle(train_files)
train_files = train_files[:N_TRAIN]

X_train = []
for f in tqdm(train_files, desc="Extracting training features"):
    feat = extract_features(f)
    if feat is not None:
        X_train.append(feat)

X_train = np.array(X_train)
print(f"✅ Training features shape: {X_train.shape}")

# -----------------------------
# 4️⃣ LOAD TEST DATA (NORMAL + ABNORMAL)
# -----------------------------
test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".wav")]
random.shuffle(test_files)
test_files = test_files[:N_TEST]

X_test, y_test = [], []
for f in tqdm(test_files, desc="Extracting test features"):
    feat = extract_features(f)
    if feat is not None:
        X_test.append(feat)
        # Label: 0 = normal, 1 = abnormal (based on filename)
        y_test.append(1 if "abnormal" in f.lower() else 0)

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"✅ Test features shape: {X_test.shape}")

# -----------------------------
# 5️⃣ TRAIN ISOLATION FOREST
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = IsolationForest(contamination=0.25, random_state=42)
model.fit(X_train_scaled)
print("✅ Model trained on normal data.")

# -----------------------------
# 6️⃣ EVALUATE PERFORMANCE
# -----------------------------
preds = model.predict(X_test_scaled)
y_pred = np.where(preds == -1, 1, 0)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["normal","abnormal"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred Normal","Pred Abnormal"],
            yticklabels=["True Normal","True Abnormal"])
plt.title("IsolationForest - FAN Machine Subset")
plt.show()

# -----------------------------
# 7️⃣ PREDICT SINGLE FILE
# -----------------------------
def predict_audio(file_path):
    feat = extract_features(file_path)
    feat_scaled = scaler.transform([feat])
    pred = model.predict(feat_scaled)
    return "⚠️ Abnormal" if pred[0] == -1 else "✅ Normal"

sample_file = random.choice(test_files)
print(f"\n🔍 Prediction for {os.path.basename(sample_file)} →", predict_audio(sample_file))
