import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
import seaborn as sns


def pacf_plot(series, lags=None):
    """
    Plot partial autocorrelation of the given time series.

    Parameters:
    series (pandas.Series): The time series to calculate partial autocorrelation for
    lags (int, optional): Number of lags to show in the plot. If None, uses a sensible default.
    """
    
    # Set figure size
    plt.figure(figsize=(12,6))
    
    # Use plot_pacf function from statsmodels to plot partial autocorrelation
    plot_pacf(series, lags=lags)
    
    # Customize title and labels
    plt.title(f"Partial Autocorrelation for {series.name}")
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    
    # Show the plot
    plt.show()



def residual_plot(y_true, y_pred):
    """
    Plot residuals of the predictions.

    This function takes the true output values and the predicted output values,
    and creates a residuals plot.

    Parameters:
    y_true (array-like): The true output values. 
                          It is an array or list of numerical values.
    y_pred (array-like): The predicted output values generated by the model. 
                          It is an array or list of numerical values.

    Returns:
    matplotlib.figure.Figure: Returns the figure of the residuals plot.
    """
    
    # Calculate residuals
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(10,6))

    # Scatter plot of predicted values vs residuals
    sns.scatterplot(x=y_pred, y=residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    # Plot a horizontal line at y = 0
    plt.axhline(y=0, color="r", linestyle="--")

    # Set plot title
    plt.title("Residuals vs. Predicted Values")

    plt.show()
