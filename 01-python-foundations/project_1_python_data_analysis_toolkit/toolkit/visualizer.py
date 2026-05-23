import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    plt.figure()
    plt.hist(df[column], bins=10)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_scatter(df, x, y):
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.title(f"Scatter Plot of {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def plot_correlation_matrix(df):
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()