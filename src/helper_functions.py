import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.model_selection import GridSearchCV
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_validate
import math


def plot_confusion_matrices(models: List[BaseEstimator], X: np.ndarray,
                            y: np.ndarray,  figsize: Tuple[int, int]) -> None:
    """
    Plot confusion matrices for a list of models using cross-validated predictions.

    Parameters:
    models (List[BaseEstimator]): A list of scikit-learn estimator instances.
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.

    Returns:
    None: The function plots confusion matrices for the provided models.
    """
    fig = plt.figure(figsize=figsize)

    for i, model in enumerate(models):
        ax = fig.add_subplot(math.ceil(len(models) / 3), 3, i + 1)
        y_pred = cross_val_predict(model, X, y, cv=5)

        title = type(model).__name__
        ax.set_title(title)

        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

def get_scores(model: BaseEstimator, X: np.ndarray,
                 y: np.ndarray, fit: bool = True) -> Dict[str, float]:
    """
    Compute performance scores on the data.

    Parameters:
    model (BaseEstimator): The machine learning model
    X (np.ndarray): The feature matrix used.
    y (np.ndarray): The target vector used.
    fit (bool): If True, the model will be fitted to the data. Default is True.

    Returns:
    Dict[str, float]: A dictionary containing accuracy, recall, precision, and f1 scores.
    """
    if fit:
        model.fit(X, y)

    model_predict = model.predict(X)

    scores = {
        "accuracy": accuracy_score(y, model_predict),
        "recall": recall_score(y, model_predict),
        "precision": precision_score(y, model_predict),
        "f1": f1_score(y, model_predict)
    }

    return scores

def cross_val_scores(model: BaseEstimator, X: np.ndarray,
                     y: np.ndarray, scoring: List[str], cv: int) -> pd.DataFrame:
    """
    Perform cross-validation on the given model and compute specified scoring metrics.

    Parameters:
    model (BaseEstimator): The machine learning model to be evaluated.
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.
    scoring (List[str]): A list of scoring methods to evaluate the model.
    cv (int): The number of cross-validation folds.

    Returns:
    pd.DataFrame: A DataFrame containing the cross-validation scores for each metric.
    """
    results = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=False)

    scores = {key: results[key] for key in results if key.startswith('test_')}
    scores = {key[5:]: scores[key] for key in scores}  # Remove 'test_' prefix from keys

    scores_df = pd.DataFrame(scores)

    return scores_df

def grid_search_results(grid: GridSearchCV, scoring: List[str],
                        main_score: str) -> pd.DataFrame:
    """
    Extract the best scores from a GridSearchCV object for specified scoring metrics.

    Parameters:
    grid (GridSearchCV): The GridSearchCV object after fitting.
    scoring (List[str]): A list of scoring metrics used in GridSearchCV.
    main_score (str): The primary scoring metric used to identify the best model.

    Returns:
    pd.DataFrame: A DataFrame containing the mean and standard deviation of test scores
    for each metric at the best model's index.
    """
    results = grid.cv_results_
    best_accuracy_idx = np.argmax(results[f'mean_test_{main_score}'])

    # Create DataFrame to store mean and std of test scores for each metric
    scores = pd.DataFrame(index=scoring, columns=["mean", "std"])

    # Populate the DataFrame with the best scores
    for score in scoring:
        scores.loc[score, "mean"] = results[f'mean_test_{score}'][best_accuracy_idx]
        scores.loc[score, "std"] = results[f'std_test_{score}'][best_accuracy_idx]

    return scores

def customize_bar(position: str, axes, 
                  values_font=12, pct=False, round_to=0) -> None:
    """
    Function, which customizes bar chart
    Takes axes object and:
        - gets rid of spines
        - modifies ticks
        - adds value above each bar
    Parameters:
        - position(str): modify the bar depending on how the
        bars are positioned: vertically or horizontally
    Return: None
    """
    # get rid of spines
    for spine in axes.spines.values():
        spine.set_visible(False)
    # modify ticklabels
    if position == "v":
        axes.set_yticks([])
        for tick in axes.get_xticklabels():
            tick.set_rotation(0)
    if position == "h":
        axes.set_xticks([])
        for tick in axes.get_yticklabels():
            tick.set_rotation(0)
    # add height value above each bar
    for bar in axes.patches:
        if bar.get_width() == 0:
            continue
        if position == "v":
            text_location = (bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 1/100*bar.get_height())
            value = bar.get_height()
            location = "center"
        elif position == "h":
            text_location = (bar.get_width(),
                             bar.get_y() + bar.get_height() / 2)
            value = bar.get_width()
            location = "left"
        value = (f"{round(value * 100, round_to)}%" if pct
                 else str(round(value, round_to)))
  
        axes.text(text_location[0],
                text_location[1],
                str(value),
                fontsize=values_font,
                ha=location)