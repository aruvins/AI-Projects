import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    
def plot_histogram(df, column, save_path=None):
    plt.figure()
    plt.hist(df[column], bins=10)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    if save_path:
        save_plot(save_path)
    
    plt.show()

def plot_scatter(df, x, y, save_path=None):
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.title(f"Scatter Plot of {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)

    if save_path:
        save_plot(save_path)
    
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    
    if save_path:
        save_plot(save_path)
    
    plt.show()
