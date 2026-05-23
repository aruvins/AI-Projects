from demos.demo_vectors import run as run_vectors
from demos.demo_transformations import run as run_transform
from demos.demo_eigenvalues import run as run_eigen

def main():
    print("Matrix & Tensor Playground")

    print("\n1. Vectors")
    run_vectors()

    print("\n2. Transformations")
    run_transform()

    print("\n3. Eigenvalues")
    run_eigen()

if __name__ == "__main__":
    main()