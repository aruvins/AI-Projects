import numpy as np

def dot_product(V1, V2):
    return np.dot(V1, V2)

def magnitude(V):
    return np.linalg.norm(V)

def projection(v, direction):
    direction = direction / np.linalg.norm(direction)
    return np.dot(v, direction) * direction

def eigen_analysis(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

