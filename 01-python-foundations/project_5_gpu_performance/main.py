import os
from benchmark import run_benchmarks
from plotting import plot_results

os.makedirs("output", exist_ok=True)

# Run CPU vs GPU benchmarks
results = run_benchmarks()

# Plot results
plot_results(results)

print("Benchmark complete. Results saved to output/")