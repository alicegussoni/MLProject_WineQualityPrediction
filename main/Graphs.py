# GRAPHS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 1. Load Data

with open('project_results_complete.json', 'r') as f:
    data = json.load(f)

lin = data["linear_models"]
ker = data["kernel_models"]

y_test_real = np.array(lin["predictions"]["y_test_real"])


# 2. Comparison

results_map = {
    "LR Linear": lin["LR_Linear"],
    "SVM Linear": lin["SVM_Linear"],
    "LR Polynomial": ker["LR_Polynomial"],
    "SVM Polynomial": ker["SVM_Polynomial"],
    "LR Gaussian": ker["LR_Gaussian"],
    "SVM Gaussian": ker["SVM_Gaussian"]
}

df_metrics = pd.DataFrame([
    {
        "model": name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }
    for name, metrics in results_map.items()
])



# table

df_table = df_metrics.copy()
df_table[["accuracy","precision","recall","f1"]] = df_table[["accuracy","precision","recall","f1"]].round(3)

best_acc = df_table["accuracy"].max()
best_model = df_table.loc[df_table["accuracy"].idxmax(), "model"]

fig, ax = plt.subplots(figsize=(11,5))
ax.axis('off')

table = ax.table(
    cellText=df_table.values,
    colLabels=df_table.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)


header_color = "#40466e"
best_row_color = "#d8f3dc"
best_acc_color = "#95d5b2"


for col in range(len(df_table.columns)):
    cell = table[(0, col)]
    cell.set_facecolor(header_color)
    cell.get_text().set_color("white")
    cell.get_text().set_weight("bold")


for i in range(len(df_table)):
    model_name = df_table.iloc[i]["model"]
    
    if model_name == best_model:
        for j in range(len(df_table.columns)):
            table[(i+1, j)].set_facecolor(best_row_color)
            table[(i+1, j)].get_text().set_weight("bold")
        
        table[(i+1, 1)].set_facecolor(best_acc_color)

plt.title(
    f"Performance Metrics Comparison\nBest Model: {best_model} (Accuracy = {best_acc:.3f})",
    fontsize=14,
    pad=20
)

plt.tight_layout()
plt.savefig("performance_table_highlighted.png", dpi=300, bbox_inches='tight')
plt.show()


ax = df_metrics.set_index("model").plot(
    kind="bar",
    figsize=(14,7),
    width=0.8
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1.1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# 2. Accuracy improvement vs baseline

baseline_acc = results_map["LR Linear"]["accuracy"]

other_models = [m for m in results_map.keys() if m != "LR Linear"]

improvements = [
    ((results_map[m]["accuracy"] - baseline_acc) / baseline_acc) * 100
    for m in other_models
]

plt.figure(figsize=(10,6))
colors = ['green' if x > 0 else 'red' for x in improvements]

plt.barh(other_models, improvements, color=colors)
plt.axvline(0, color='black')
plt.xlabel("Accuracy Improvement (%)")
plt.title("Improvement over LR Linear")
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# 3. ROC Curves

def plot_roc_curve(y_true, scores, label, ax):

    y_true = np.array(y_true)
    scores = np.array(scores)

    if scores.min() < 0 or scores.max() > 1:
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    thresholds = np.linspace(0,1,200)
    tpr = []
    fpr = []

    for t in thresholds:
        y_pred = np.where(scores >= t, 1, -1)

        tp = np.sum((y_pred==1)&(y_true==1))
        fp = np.sum((y_pred==1)&(y_true==-1))
        fn = np.sum((y_pred==-1)&(y_true==1))
        tn = np.sum((y_pred==-1)&(y_true==-1))

        tpr.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        fpr.append(fp/(fp+tn) if (fp+tn)>0 else 0)

    auc = np.abs(np.trapz(tpr, fpr))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")


plt.figure(figsize=(9,7))
ax = plt.gca()

# Linear
plot_roc_curve(y_test_real, lin["predictions"]["y_score_linear_lr"], "LR Linear", ax)
plot_roc_curve(y_test_real, lin["predictions"]["y_score_linear_svm"], "SVM Linear", ax)

# Polynomial
plot_roc_curve(y_test_real, ker["predictions"]["y_prob_polynomial_lr"], "LR Polynomial", ax)
plot_roc_curve(y_test_real, ker["predictions"]["y_score_polynomial_svm"], "SVM Polynomial", ax)

# Gaussian
plot_roc_curve(y_test_real, ker["predictions"]["y_prob_gaussian_lr"], "LR Gaussian", ax)
plot_roc_curve(y_test_real, ker["predictions"]["y_score_gaussian_svm"], "SVM Gaussian", ax)

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 4. Learning Curves

def plot_all_learning_curves():

    models_info = [
        ("LR Linear", lin["LR_Linear"]),
        ("SVM Linear", lin["SVM_Linear"]),
        ("LR Polynomial", ker["LR_Polynomial"]),
        ("SVM Polynomial", ker["SVM_Polynomial"]),
        ("LR Gaussian", ker["LR_Gaussian"]),
        ("SVM Gaussian", ker["SVM_Gaussian"])
    ]

    fig, axes = plt.subplots(3,2, figsize=(14,14))
    axes = axes.ravel()

    for i,(name,model) in enumerate(models_info):

        train_loss = model["loss_history"]
        val_loss = model.get("val_loss_history",None)

        axes[i].plot(train_loss, label="Train", linewidth=2)

        if val_loss is not None:
            axes[i].plot(val_loss, linestyle="--", label="Validation", linewidth=2)

        axes[i].set_title(name)
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel("Loss")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    fig.suptitle("Learning Curves - All Models", fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

plot_all_learning_curves()

# 5. Confusion Matrix

prediction_sources = {
    "LR Linear": np.array(lin["predictions"]["y_pred_linear_lr"]),
    "SVM Linear": np.array(lin["predictions"]["y_pred_linear_svm"]),
    "LR Polynomial": np.where(np.array(ker["predictions"]["y_prob_polynomial_lr"])>=0.5,1,-1),
    "SVM Polynomial": np.array(ker["predictions"]["y_pred_polynomial_svm"]),
    "LR Gaussian": np.where(np.array(ker["predictions"]["y_prob_gaussian_lr"])>=0.5,1,-1),
    "SVM Gaussian": np.array(ker["predictions"]["y_pred_gaussian_svm"])
}

def get_confusion_matrix(y_true,y_pred):
    tp = np.sum((y_pred==1)&(y_true==1))
    tn = np.sum((y_pred==-1)&(y_true==-1))
    fp = np.sum((y_pred==1)&(y_true==-1))
    fn = np.sum((y_pred==-1)&(y_true==1))
    return np.array([[tp,fn],[fp,tn]])

fig, axes = plt.subplots(3,2, figsize=(12,14))
axes = axes.ravel()

for i,(name,y_pred) in enumerate(prediction_sources.items()):
    cm = get_confusion_matrix(y_test_real,y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        ax=axes[i],
        xticklabels=["Pred +1","Pred -1"],
        yticklabels=["True +1","True -1"],
        cbar=False
    )

    axes[i].set_title(name)

plt.tight_layout()
plt.show()


# 6. Misclassification Analysis
error_sets = {}

for name, y_pred in prediction_sources.items():
    errors = set(np.where(y_pred != y_test_real)[0])
    error_sets[name] = errors
    print(f"{name} - Errors number: {len(errors)}")

baseline_errors = error_sets["LR Linear"]

for name, errors in error_sets.items():
    if name != "LR Linear":
        corrected = baseline_errors - errors
        print(f"{name} correct {len(corrected)}errors of LR Linear")

