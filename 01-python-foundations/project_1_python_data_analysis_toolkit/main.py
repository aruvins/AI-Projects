from toolkit.loader import load_csv
from toolkit.stats import basic_stats, correlations, column_stats
from toolkit.visualizer import plot_histogram, plot_scatter, plot_correlation_matrix
from toolkit.report import generate_report, save_report

def main():
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
    plot_histogram(df, "age")
    plot_scatter(df, "age", "salary")
    plot_correlation_matrix(df)

    # Report
    report = generate_report(df, stats, corr)
    save_report(report)

if __name__ == "__main__":
    main()