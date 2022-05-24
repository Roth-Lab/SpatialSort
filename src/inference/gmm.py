from sklearn.mixture import GaussianMixture


def inference_gmm(k_clusters, data, seed=None):
    """
        Returning labels, means, and covariances using GMM
    """
    # Fit using a GMM
    fitted_gmm = GaussianMixture(n_components=k_clusters, n_init=5, random_state=seed).fit(data)
    predicted_labels = fitted_gmm.predict(data)
    means = fitted_gmm.means_
    covs = fitted_gmm.covariances_

    return predicted_labels, means, covs
