import numpy as np
def pca(X, k):
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    norm_X = X - mean
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    data = np.dot(norm_X, np.transpose(feature))
    return data
def main():
    X = np.array([[-1, 10], [-3, -4], [-3, -2], [1, 2], [2, 4], [3, 1]])
    print(pca(X,1))
main()
