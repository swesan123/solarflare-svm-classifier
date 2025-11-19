"""Main driver for Assignment 2: load features, evaluate SVMs and save results.

This module provides:
- load_features: assemble feature matrices from dataset folders (pos/neg .npy files)
- preprocess: simple NaN handling and scaling
- evaluate_svm: 5-fold stratified evaluation returning mean/std of TSS
- plotting helpers for bar charts and confusion matrices
- main: example driver that runs all feature combinations and writes results to results/

Usage:
    python main.py

Note:
    Update dataset paths in main() to point to your local data folders.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
# use non-interactive backend so saving figures works on headless systems
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


feature_sets = ['FSI', 'FSII', 'FSIII', 'FSIV']

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def load_features(dataset_path, feature_sets):
    """Load and assemble features for positive and negative classes.

    Parameters
    ----------
    dataset_path : str
        Path to a folder containing pos_/neg_ .npy feature files.
    feature_sets : sequence of str
        Requested feature sets, subset of {'FSI','FSII','FSIII','FSIV'}.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix with negative rows followed by positive rows.
    y : np.ndarray, shape (n_samples,)
        Label vector with 0 for negatives and 1 for positives.
    """
    X_pos, X_neg = [], []

    if 'FSI' in feature_sets or 'FSII' in feature_sets:
        data_pos = np.load(os.path.join(dataset_path,'pos_features_main_timechange.npy'))
        data_neg = np.load(os.path.join(dataset_path,'neg_features_main_timechange.npy'))
        if 'FSI' in feature_sets:
            X_pos.append(data_pos[:, :18])
            X_neg.append(data_neg[:, :18])
        if 'FSII' in feature_sets:
            X_pos.append(data_pos[:, 18:90])
            X_neg.append(data_neg[:, 18:90])
    if 'FSIII' in feature_sets:
        X_pos.append(np.load(os.path.join(dataset_path,'pos_features_historical.npy')))
        X_neg.append(np.load(os.path.join(dataset_path,'neg_features_historical.npy')))
    if 'FSIV' in feature_sets:
        X_pos.append(np.load(os.path.join(dataset_path,'pos_features_maxmin.npy')))
        X_neg.append(np.load(os.path.join(dataset_path,'neg_features_maxmin.npy')))      

    X_pos = np.concatenate(X_pos, axis=1)
    X_neg = np.concatenate(X_neg, axis=1)

    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_pos))])

    return X, y

def preprocess(X):
    """Replace NaNs and scale features.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled matrix with zero mean and unit variance (per feature).
    """
    X = np.nan_to_num(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def true_skill_score(cm):
    """Compute True Skill Score (TSS) from a confusion matrix.

    Parameters
    ----------
    cm : array-like, shape (2,2)
        Confusion matrix with layout [[TN, FP], [FN, TP]].

    Returns
    -------
    float
        TSS score in range [-1, 1].
    """
    TP, FN, FP, TN = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    return (TP/(TP+FN)) - (FP/(FP+TN))

def evaluate_svm(X, y, C=1.0, gamma='scale', kernel='rbf'):
    """Evaluate an SVM with 5-fold stratified CV and return mean/std TSS.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels.
    C, gamma, kernel : svm hyperparameters passed to sklearn.svm.SVC

    Returns
    -------
    mean_tss, std_tss : tuple(float, float)
        Mean and standard deviation of TSS across folds.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tss_scores = []

    for train_idx, test_idx in kf.split(X, y):
        clf = svm.SVC(C=C, gamma=gamma, kernel=kernel)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        cm = confusion_matrix(y[test_idx], y_pred)
        tss_scores.append(true_skill_score(cm))
    return np.mean(tss_scores), np.std(tss_scores)

def generate_feature_combinations(features):
    """Generate all non-empty combinations of the given feature set list.

    Parameters
    ----------
    features : sequence of str

    Returns
    -------
    list[list[str]]
        List of feature-set combinations (each a list of strings).
    """
    combos = [[]]

    for f in features:
        new_combos = []
        for c in combos:
            new_combos.append(c + [f])
        combos.extend(new_combos)
    
    combos = [c for c in combos if c]
    return combos

def plot_bar_chart(df, title, save_path):
    """Plot a bar chart of mean TSS with error bars and annotated values.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'Feature Set', 'Mean TSS', and 'Std Dev'.
    title : str
        Plot title.
    save_path : str
        File path to save the PNG image.
    """
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x='Feature Set', y='Mean TSS', data=df, errorbar=None)
    x = np.arange(len(df))
    plt.errorbar(x, df['Mean TSS'].values, yerr=df['Std Dev'].values, fmt='none', c='black', capsize=5)

    # annotate each bar with its value (two decimal places)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(p.get_x() + p.get_width() / 2, h),
                    xytext=(0, 3),               # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8)
        
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xticks(ticks=x, labels=df['Feature Set'], rotation=45, ha='right')
    plt.ylabel('TSS')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(dataset_path, all_combos, save_path):
    """Produce a grid of confusion matrices, one per feature-combo.

    Parameters
    ----------
    dataset_path : str
        Path to folder containing the feature .npy files.
    all_combos : sequence of sequence of str
        List of feature combinations to evaluate.
    save_path : str
        Output file path for the figure.
    """
    fig, axes = plt.subplots(3,5, figsize=(20, 12))
    axes = axes.flatten()

    for i, combo in enumerate(all_combos):
        X, y = load_features(dataset_path, combo)
        X = preprocess(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        clf = svm.SVC(C=1.0, gamma='scale', kernel='rbf')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Neg", "Pos"],
            cmap="Blues",
            values_format='d',
            ax=axes[i]

        )

        disp.ax_.set_title('+'.join(combo), fontsize=9)
        disp.ax_.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def main():
    """Top-level script: run evaluations for all feature combinations.

    - Ensures results dir exists.
    - Loads datasets (update paths inside this function if needed).
    - Evaluates each combo, saves CSVs and PNGs to RESULTS_DIR.
    """
    # ensure results folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Change path here
    dataset_2010_path = '/home/swesan/Year-4-Notes/4AL3-Machine-Learning/Assignment2_test/data-2010-15'
    dataset_2020_path = '/home/swesan/Year-4-Notes/4AL3-Machine-Learning/Assignment2_test/data-2020-24'
    all_combos = generate_feature_combinations(feature_sets)

    results_2010 = []
    results_2020 = []

    for combo in all_combos:
        X_2010, y_2010 = load_features(dataset_2010_path, combo)
        X_2020, y_2020 = load_features(dataset_2020_path, combo)

        X_2010 = preprocess(X_2010)
        X_2020 = preprocess(X_2020)

        mean_tss_2010, std_tss_2010 = evaluate_svm(X_2010, y_2010)
        mean_tss_2020, std_tss_2020 = evaluate_svm(X_2020, y_2020)

        combo_name = "+".join(combo)

        results_2010.append({'Feature Set': combo_name, 'Mean TSS': mean_tss_2010, 'Std Dev': std_tss_2010})
        results_2020.append({'Feature Set': combo_name, 'Mean TSS': mean_tss_2020, 'Std Dev': std_tss_2020})

        print(f"{combo} -> mean={mean_tss_2010:.3f}, std={std_tss_2010:.3f}\n")
        print(f"{combo} -> mean={mean_tss_2020:.3f}, std={std_tss_2020:.3f}\n")
    
    df_2010 = pd.DataFrame(results_2010)
    df_2020 = pd.DataFrame(results_2020)

    # save into results folder
    df_2010.to_csv(os.path.join(RESULTS_DIR, 'tss_results_2010.csv'), index=False)
    df_2020.to_csv(os.path.join(RESULTS_DIR, 'tss_results_2020.csv'), index=False)

    plot_bar_chart(df_2010, 'SVM Performance 2010', os.path.join(RESULTS_DIR, 'tss_bar_2010.png'))
    plot_bar_chart(df_2020, 'SVM Performance 2020', os.path.join(RESULTS_DIR, 'tss_bar_2020.png'))

    plot_confusion_matrix(dataset_2010_path, all_combos, os.path.join(RESULTS_DIR, 'confusion_matrix_2010.png'))
    plot_confusion_matrix(dataset_2020_path, all_combos, os.path.join(RESULTS_DIR, 'confusion_matrix_2020.png'))


if __name__ == '__main__':
    main()
