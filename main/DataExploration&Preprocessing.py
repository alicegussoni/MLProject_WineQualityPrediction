# Data exploration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from Load_Data import load_data

df = load_data("data/winequality-red.csv",
               "data/winequality-white.csv")

print("Dataset shape (rows, columns):", df.shape)
print(df.info())
print("Summary statistics:\n", df.describe())
print("Missing values:\n", df.isnull().sum())


# 2. UNIVARIATE ANALYSIS
numerical_cols = df.select_dtypes(include=np.number).columns

# Histograms
df[numerical_cols].hist(bins=30, figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(col)
    axes[i].set_xlabel('')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# 3. CLASS DISTRIBUTION
df['is_good'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

def add_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold')

plt.figure(figsize=(10, 6))
ax1 = sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Original Dataset Class Distribution')
add_labels(ax1)
plt.show()

plt.figure(figsize=(8, 6))
ax2 = sns.countplot(x='is_good', data=df, palette='coolwarm')
plt.title('Binary Class Distribution (Good >= 6)')
plt.xticks([0, 1], ['Bad (<6)', 'Good (>=6)'])
add_labels(ax2)
plt.show()

# Quality distribution by wine type
sns.countplot(data=df, x='quality', hue='wine_type',
              palette={'red': '#8B0000', 'white': '#F5F5DC'})
plt.title('Quality Distribution: Red vs White Wines')
plt.show()


# 4. BIVARIATE ANALYSIS
target = "quality"

df[numerical_cols].corr()[target].sort_values(ascending=False)

cols_to_plot = [col for col in numerical_cols if col != target]
charts_per_page = 4
n_pages = math.ceil(len(cols_to_plot) / charts_per_page)

for p in range(n_pages):
    start = p * charts_per_page
    end = start + charts_per_page
    subset_cols = cols_to_plot[start:end]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Scatterplot: Variables vs {target} - Group {p+1}',
                 fontsize=18)
    axes = axes.flatten()

    for i, col in enumerate(subset_cols):
        sns.regplot(data=df, x=col, y=target, ax=axes[i],
                    scatter_kws={'alpha': 0.3, 'color': 'teal'},
                    line_kws={'color': 'red'})
        axes[i].set_title(f"{col} vs {target}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Boxplots vs target
for p in range(n_pages):
    start = p * charts_per_page
    end = start + charts_per_page
    subset_cols = cols_to_plot[start:end]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Variable Analysis - Group {p+1}', fontsize=18)
    axes = axes.flatten()

    for i, col in enumerate(subset_cols):
        sns.boxplot(data=df, x=target, y=col,
                    ax=axes[i], palette='viridis')
        axes[i].set_title(f"{col} distribution by {target}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# 5. OUTLIER REMOVAL (IQR METHOD)

def remove_outliers_iqr(df, columns):
    df_final = df.copy()
    for col in columns:
        Q1 = df_final[col].quantile(0.25)
        Q3 = df_final[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_final = df_final[
            (df_final[col] >= lower_bound) &
            (df_final[col] <= upper_bound)
        ]
    return df_final


cols_with_outliers = [
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide'
]

df_clean = remove_outliers_iqr(df, cols_with_outliers)

print(f"Rows before: {len(df)}, Rows after: {len(df_clean)}")



# 6. CLASS DISTRIBUTION AFTER OUTLIER REMOVAL

df_clean['is_good'] = df_clean['quality'].apply(lambda x: 1 if x >= 6 else 0)

plt.figure(figsize=(10, 6))
ax1 = sns.countplot(x='quality', data=df_clean, palette='viridis')
plt.title('Class Distribution After Outlier Removal')
add_labels(ax1)
plt.show()

plt.figure(figsize=(8, 6))
ax2 = sns.countplot(x='is_good', data=df_clean, palette='coolwarm')
plt.title('Binary Class Distribution After Outlier Removal')
plt.xticks([0, 1], ['Bad (<6)', 'Good (>=6)'])
add_labels(ax2)
plt.show()

# Correlation heatmap
numeric_df = df_clean.select_dtypes(include='number')
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm")
plt.show()


df_clean["target"] = (df_clean['quality'] >= 6).astype(int)

# Pairplot
sns.pairplot(df_clean[['alcohol',
                       'volatile acidity',
                       'density',
                       'target']],
             hue='target')
plt.show()

# Boxplot of alcohol vs binary target
sns.boxplot(data=df_clean, x='target', y='alcohol')
plt.show()



# 7. FEATURE PREPARATION

X = df_clean.drop(columns=['quality', 'target', 'is_good'],
                  errors='ignore')
y = df_clean['target']

X['wine_type'] = X['wine_type'].map({'red': 0, 'white': 1})


# 8. TRAIN-TEST SPLIT (stratified)

red_index = np.where(X['wine_type'] == 0)[0]
white_index = np.where(X['wine_type'] == 1)[0]

random_gen = np.random.default_rng(seed=42)
random_gen.shuffle(red_index)
random_gen.shuffle(white_index)

test_size = 0.2

n_red_test = int(len(red_index) * test_size)
n_white_test = int(len(white_index) * test_size)

test_index = np.concatenate([
    red_index[:n_red_test],
    white_index[:n_white_test]
])

train_index = np.concatenate([
    red_index[n_red_test:],
    white_index[n_white_test:]
])

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("Train set proportions:")
print(pd.Series(X['wine_type'].iloc[train_index])
      .value_counts(normalize=True))

print("\nTest set proportions:")
print(pd.Series(X['wine_type'].iloc[test_index])
      .value_counts(normalize=True))



# 9. FEATURE STANDARDIZATION

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)

X_train_scaled = (X_train - mu) / sigma
X_test_scaled = (X_test - mu) / sigma



# 10. SAVE PROCESSED DATA

X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Test class distribution:\n",
      y_test.value_counts(normalize=True))