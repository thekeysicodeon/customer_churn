"""
Intelligent Customer Churn Prediction System
ANN-Based Churn Prediction | Soft Computing Course
Dataset: Telco Customer Churn (Kaggle)
Models: Custom Backprop ANN + Keras ANN + LR + RF + SVM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────

def load_and_preprocess(path='telco_customer_churn.csv'):
    df = pd.read_csv(path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    multi_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cat, drop_first=True)

    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values
    feature_names = df.drop('Churn', axis=1).columns.tolist()
    return X, y, feature_names


# ─────────────────────────────────────────────
# 2. Custom ANN (NumPy Backpropagation)
# ─────────────────────────────────────────────

class CustomANN:
    """
    Feedforward ANN with manual backpropagation.
    Architecture: Input → Hidden (ReLU + Dropout) → Output (Sigmoid)
    Optimizer: Mini-batch SGD with momentum
    Loss: Binary Cross-Entropy
    """

    def __init__(self, layer_sizes, learning_rate=0.005, momentum=0.9,
                 epochs=200, batch_size=64, dropout_rate=0.2):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.train_losses = []
        self.val_losses = []
        self._init_weights()

    def _init_weights(self):
        self.W, self.b = [], []
        self.vW, self.vb = [], []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            W = np.random.randn(fan_in, self.layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.W.append(W); self.b.append(b)
            self.vW.append(np.zeros_like(W)); self.vb.append(np.zeros_like(b))

    @staticmethod
    def relu(z):       return np.maximum(0, z)
    @staticmethod
    def relu_d(z):     return (z > 0).astype(float)
    @staticmethod
    def sigmoid(z):    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X, training=True):
        self.A = [X]; self.Z = []; self.masks = []
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b; self.Z.append(z)
            if i < len(self.W) - 1:
                a = self.relu(z)
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    a *= mask; self.masks.append(mask)
                else:
                    self.masks.append(None)
            else:
                a = self.sigmoid(z)
            self.A.append(a)
        return a

    def _bce(self, yt, yp):
        eps = 1e-9
        return -np.mean(yt * np.log(yp + eps) + (1 - yt) * np.log(1 - yp + eps))

    def _backward(self, X, y):
        m = X.shape[0]
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        dA = self.A[-1] - y.reshape(-1, 1)
        for i in reversed(range(len(self.W))):
            dZ = dA if i == len(self.W) - 1 else dA * self.relu_d(self.Z[i])
            if i < len(self.W) - 1 and self.masks[i] is not None:
                dZ *= self.masks[i]
            gW[i] = self.A[i].T @ dZ / m
            gb[i] = np.sum(dZ, axis=0, keepdims=True) / m
            dA = dZ @ self.W[i].T
        for i in range(len(self.W)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * gW[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * gb[i]
            self.W[i] += self.vW[i]; self.b[i] += self.vb[i]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for epoch in range(self.epochs):
            idx = np.random.permutation(len(X_train))
            Xs, ys = X_train[idx], y_train[idx]
            for s in range(0, len(X_train), self.batch_size):
                self._forward(Xs[s:s+self.batch_size], training=True)
                self._backward(Xs[s:s+self.batch_size], ys[s:s+self.batch_size])
            yp = self._forward(X_train, training=False).flatten()
            self.train_losses.append(self._bce(y_train, yp))
            if X_val is not None:
                yv = self._forward(X_val, training=False).flatten()
                self.val_losses.append(self._bce(y_val, yv))
            if (epoch + 1) % 25 == 0:
                msg = f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {self.train_losses[-1]:.4f}"
                if X_val is not None:
                    msg += f" | Val Loss: {self.val_losses[-1]:.4f}"
                print(msg)

    def predict_proba(self, X): return self._forward(X, training=False).flatten()
    def predict(self, X, thr=0.5): return (self.predict_proba(X) >= thr).astype(int)


# ─────────────────────────────────────────────
# 3. Keras Model Builder for Tuner
# ─────────────────────────────────────────────

def build_keras_model(hp, n_features):
    model = Sequential([
        Dense(hp.Int('units_1', 32, 128, step=32), activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(hp.Float('dropout_1', 0.1, 0.4, step=0.1)),
        Dense(hp.Int('units_2', 16, 64, step=16), activation='relu'),
        BatchNormalization(),
        Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [0.001, 0.005, 0.01])),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


# ─────────────────────────────────────────────
# 4. Metrics Helper
# ─────────────────────────────────────────────

def get_metrics(yt, yp, yprob):
    return {
        'Accuracy':  round(accuracy_score(yt, yp) * 100, 2),
        'Precision': round(precision_score(yt, yp) * 100, 2),
        'Recall':    round(recall_score(yt, yp) * 100, 2),
        'F1-Score':  round(f1_score(yt, yp) * 100, 2),
        'ROC-AUC':   round(roc_auc_score(yt, yprob) * 100, 2),
    }


# ─────────────────────────────────────────────
# 5. Main Pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Intelligent Customer Churn Prediction System")
    print("=" * 60)

    # --- Load ---
    X, y, feature_names = load_and_preprocess()
    n_features = X.shape[1]
    print(f"\nFeatures: {n_features} | Samples: {len(y)}")

    # --- Split & Scale ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- SMOTE ---
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train_sc, y_train)
    print(f"After SMOTE: {len(y_sm)} samples")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sm, y_sm, test_size=0.15, random_state=42, stratify=y_sm)

    # --- Custom ANN ---
    print("\n[1] Training Custom ANN (NumPy Backpropagation)...")
    cann = CustomANN([n_features, 64, 32, 1], learning_rate=0.005,
                     momentum=0.9, epochs=200, batch_size=64, dropout_rate=0.2)
    cann.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    yp_cann  = cann.predict(X_test_sc)
    ypr_cann = cann.predict_proba(X_test_sc)

    # --- Keras ANN ---
    print("\n[2] Keras Hyperparameter Tuning...")
    tuner = kt.RandomSearch(
        lambda hp: build_keras_model(hp, n_features),
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=10, executions_per_trial=1,
        directory='kt_churn', project_name='churn_ann', overwrite=True
    )
    tuner.search(X_sm, y_sm, epochs=50, validation_split=0.15,
                 callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max')],
                 verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    keras_model = tuner.hypermodel.build(best_hp)
    history = keras_model.fit(
        X_sm, y_sm, epochs=100, batch_size=64, validation_split=0.15,
        callbacks=[EarlyStopping(monitor='val_auc', patience=10, mode='max',
                                 restore_best_weights=True),
                   ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)],
        verbose=0
    )
    yp_keras  = (keras_model.predict(X_test_sc) >= 0.5).astype(int).flatten()
    ypr_keras = keras_model.predict(X_test_sc).flatten()

    # --- Logistic Regression ---
    print("\n[3] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_sm, y_sm)
    yp_lr  = lr.predict(X_test_sc)
    ypr_lr = lr.predict_proba(X_test_sc)[:, 1]

    # --- Random Forest ---
    print("\n[4] Random Forest (GridSearchCV)...")
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        cv=3, scoring='roc_auc', n_jobs=-1
    )
    rf_gs.fit(X_sm, y_sm)
    rf = rf_gs.best_estimator_
    yp_rf  = rf.predict(X_test_sc)
    ypr_rf = rf.predict_proba(X_test_sc)[:, 1]

    # --- SVM ---
    print("\n[5] SVM (RBF)...")
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_sm, y_sm)
    yp_svm  = svm.predict(X_test_sc)
    ypr_svm = svm.predict_proba(X_test_sc)[:, 1]

    # --- Results ---
    results = {
        'Custom ANN':    get_metrics(y_test, yp_cann, ypr_cann),
        'Keras ANN':     get_metrics(y_test, yp_keras, ypr_keras),
        'Logistic Reg.': get_metrics(y_test, yp_lr, ypr_lr),
        'Random Forest': get_metrics(y_test, yp_rf, ypr_rf),
        'SVM':           get_metrics(y_test, yp_svm, ypr_svm),
    }
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 60)
    print(" Final Model Comparison")
    print("=" * 60)
    print(results_df.to_string())

    # --- Plots ---
    _plot_roc(y_test, [('Custom ANN', ypr_cann), ('Keras ANN', ypr_keras),
                       ('Logistic Reg.', ypr_lr), ('Random Forest', ypr_rf), ('SVM', ypr_svm)])
    _plot_cm(y_test, [('Custom ANN', yp_cann), ('Keras ANN', yp_keras),
                      ('Logistic Reg.', yp_lr), ('Random Forest', yp_rf), ('SVM', yp_svm)])
    _plot_comparison(results_df)


def _plot_roc(y_test, models):
    plt.figure(figsize=(9, 7))
    for name, prob in models:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves', fontsize=14, fontweight='bold')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig('roc_curves.png', dpi=150); plt.show()


def _plot_cm(y_test, models):
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    for ax, (name, pred) in zip(axes, models):
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', ax=ax,
                    cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax.set_title(name, fontweight='bold')
    plt.tight_layout(); plt.savefig('confusion_matrices.png', dpi=150); plt.show()


def _plot_comparison(df):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df)); w = 0.15
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    for i, (col, c) in enumerate(zip(df.columns, colors)):
        ax.bar(x + i * w, df[col], w, label=col, color=c, alpha=0.85)
    ax.set_xticks(x + w * 2); ax.set_xticklabels(df.index, rotation=15)
    ax.set_ylim(50, 100); ax.set_ylabel('Score (%)'); ax.legend()
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig('model_comparison.png', dpi=150); plt.show()


if __name__ == '__main__':
    main()
