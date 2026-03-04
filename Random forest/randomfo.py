# ----------------------------------------
# Random Forest on Breast Cancer Dataset
# ----------------------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

# ----------------------------------------
# 1️⃣ Load Dataset
# ----------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

print("\nDataset Loaded Successfully")
print("Classes:", class_names)
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

# ----------------------------------------
# 2️⃣ Save Dataset as CSV
# ----------------------------------------
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["target"] = df["target"].map({0: "malignant", 1: "benign"})

df.to_csv("breast_cancer_dataset.csv", index=False)
print("\nDataset saved as 'breast_cancer_dataset.csv'")

# ----------------------------------------
# 3️⃣ Build Random Forest Model
# ----------------------------------------
model = RandomForestClassifier(
    n_estimators=5,
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model.fit(X, y)
print("\nRandom Forest built using ENTROPY")

# ----------------------------------------
# 4️⃣ Model Accuracy
# ----------------------------------------
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ----------------------------------------
# 5️⃣ Confusion Matrix
# ----------------------------------------
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ----------------------------------------
# 6️⃣ Save All Decision Trees to PDF
# ----------------------------------------
with PdfPages("RandomForest_Trees.pdf") as pdf:
    for i, tree in enumerate(model.estimators_):
        plt.figure(figsize=(15, 8))

        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True
        )

        plt.title(f"Decision Tree {i+1}")
        pdf.savefig()
        plt.close()

print("\nAll trees saved in 'RandomForest_Trees.pdf'")

# ----------------------------------------
# Tree-wise Predictions (Majority Voting Demo)
# ----------------------------------------

new_sample = X[0].reshape(1, -1)

print("\nTree-wise Predictions:\n")

tree_predictions = []

for i, tree in enumerate(model.estimators_):
    pred = tree.predict(new_sample)[0]   # VERY IMPORTANT [0]
    decoded = class_names[int(pred)]     # Convert to integer
    tree_predictions.append(decoded)
    print(f"Tree {i+1} Prediction: {decoded}")

# Majority Voting
final_vote = max(set(tree_predictions), key=tree_predictions.count)
print("\nFinal Prediction (Majority Voting):", final_vote)
