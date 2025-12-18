
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# This line ensures that your results are reproducible and consistent every time.
torch.manual_seed(42)
import matplotlib.pyplot as plt

def plot_numpyarr(input_arr, output_arr):
    plt.figure(figsize=(5, 3))
    plt.scatter(input_arr, output_arr, color='blue', label='Data Points')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.title('Input vs Output')
    plt.legend()
    plt.show()



def plot_regression_line(x, y_true, y_pred, title="Regression Plot"):
    """
    Plots the regression line (prediction) against the original input/output scatter plot.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Create the plot
    plt.figure(figsize=(5, 3))
    
    # Plot original data
    plt.scatter(x, y_true, color='blue', alpha=0.5, label='Original Data')
    
    # Plot regression line
    # Sort x values to ensure the line plot is connected correctly if data isn't sorted
    sorted_indices = x.argsort(axis=0).flatten()
    plt.plot(x[sorted_indices], y_pred[sorted_indices], color='red', linewidth=2, label='Model Prediction')
    
    plt.title(title)
    plt.xlabel("Input (Normalized)")
    plt.ylabel("Output (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()

def dynamic_plot(model, x, y_true, y_pred, epoch, loss):
    """
    Evaluates the model and calls plot_regression_line to visualize performance.
    """
    # Switch model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Generate predictions
        y_pred = model(x)
    
    # Generate title with training info
    plot_title = f'Epoch [{epoch}], Loss: {loss:.4f}'
    
    # Call the plotting function
    # Optional: Use clear_output(wait=True) here if you want an animation effect 
    # instead of a long list of plots.
    plot_regression_line(x, y_true, y_pred, title=plot_title)
    
    # Switch model back to training mode
    model.train()