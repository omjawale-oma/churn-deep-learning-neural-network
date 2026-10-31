"""Customer churn prediction with a neural network (Keras).

Uses the same telecom dataset as the Power BI churn analysis project.
Put telco_churn_unclean.csv inside the data/ folder, then run:  python churn_nn.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, recall_score, precision_score,
                             accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)

# ---------- 1. Load and clean ----------
df = pd.read_csv("data/telco_churn_unclean.csv")
print("Raw shape:", df.shape)
print("Columns:", list(df.columns))

df = df.drop_duplicates()
df = df.drop(columns=["customerID"], errors="ignore")

# MonthlyCharges: text with $ signs -> number
df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"].astype(str).str.replace("$", "", regex=False).str.strip(),
    errors="coerce")
# TotalCharges: blanks / non-numeric -> number, fill with tenure * monthly
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

# Standardize text columns (e.g. "month to month" vs "Month-to-month")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()
df["Contract"] = (df["Contract"].str.lower().str.replace(" ", "-")
                  .map({"month-to-month": "Month-to-month",
                        "one-year": "One year",
                        "two-year": "Two year"})
                  .fillna(df["Contract"]))

df = df.dropna(subset=["MonthlyCharges"])
print("Clean shape:", df.shape)

# ---------- 2. Features and target ----------
y = df["Churn"].str.lower().map({"yes": 1, "no": 0})
mask = y.notna()
df, y = df[mask], y[mask].astype(int)
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True).astype("float32")
print("Churn rate: %.2f%%" % (100 * y.mean()))

# ---------- 3. Split and scale ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 4. Neural network ----------
w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: w[0], 1: w[1]}

model = keras.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

history = model.fit(
    X_train_s, y_train, validation_split=0.2, epochs=100, batch_size=32,
    class_weight=class_weight, verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)])

nn_prob = model.predict(X_test_s).ravel()
nn_pred = (nn_prob > 0.5).astype(int)
print("\n=== Neural network ===")
print(classification_report(y_test, nn_pred, target_names=["Stayed", "Churned"]))
print("Confusion matrix:\n", confusion_matrix(y_test, nn_pred))

# ---------- 5. Baselines for comparison ----------
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=SEED).fit(X_train, y_train)
models = {
    "Neural network": (nn_pred, nn_prob),
    "Logistic regression": (lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1]),
    "Random forest": (rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
}
rows = []
for name, (pred, prob) in models.items():
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC-AUC": roc_auc_score(y_test, prob)})
results = pd.DataFrame(rows).round(3)
print("\n=== Model comparison ===")
print(results.to_string(index=False))
results.to_csv("results.csv", index=False)

# ---------- 6. Plots ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")

model.save("churn_model.keras")
print("\nSaved: results.csv, training_curves.png, churn_model.keras")
