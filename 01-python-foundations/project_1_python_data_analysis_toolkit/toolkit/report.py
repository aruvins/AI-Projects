def generate_report(df, stats, corr):
    report = []
    report.append("=== DATA ANALYSIS REPORT ===\n")
    report.append(f"Dataset Shape: {df.shape}\n")

    report.append("=== BASIC STATISTICS ===")
    report.append(str(stats))

    report.append("\n=== CORRELATION MATRIX ===")
    report.append(str(corr))

    return "\n".join(report)

def save_report(report: str, filename="outputs/report.txt"):
    with open(filename, "w") as f:
        f.write(report)

    print(f"[INFO] Report saved to {filename}")
