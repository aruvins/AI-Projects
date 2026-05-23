import os

from toolkit.loader import load_csv
from toolkit.stats import basic_stats, correlations, column_stats
from toolkit.visualizer import plot_histogram, plot_scatter, plot_correlation_matrix
from toolkit.report import generate_report, save_report

def main():
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots"
    # Load data
    df = load_csv("data/sample.csv")

    if df is None:
        return

    # Stats
    stats = basic_stats(df)
    corr = correlations(df)

    print("\n", stats)

    # Column analysis example
    print("\nAge stats:", column_stats(df, "age"))

    # Visualizations
    plot_histogram(df, "age", save_path=f"{save_path}/age_histogram.png")
    plot_scatter(df, "age", "salary", save_path=f"{save_path}/age_salary_scatter.png")
    plot_correlation_matrix(df, save_path=f"{save_path}/correlation_matrix.png")

    # Report
    report = generate_report(df, stats, corr)
    save_report(report)

if __name__ == "__main__":
    main()