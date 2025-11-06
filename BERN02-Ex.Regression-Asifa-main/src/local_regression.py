import numpy as np
from sklearn.linear_model import LinearRegression

def tricube(u):
    """Tricube kernel weights."""
    u = np.asarray(u)
    out = (1 - np.abs(u) ** 3) ** 3
    out[np.abs(u) >= 1] = 0
    return out

def local_simple_regression(y, x, k, x0):
    """
    Local simple regression using tricube weights.

    Parameters
    ----------
    y : array-like
        Response variable.
    x : array-like
        Predictor variable.
    k : int
        Number of neighbors.
    x0 : float
        Point where we want to estimate regression.

    Returns
    -------
    y_hat : float
        Predicted mean at x0.
    se : float
        Standard error of expected value at x0.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # distances from x0
    distances = np.abs(x.flatten() - x0)

    # find k-nearest neighbors
    idx = np.argsort(distances)[:k]
    x_nn, y_nn, d_nn = x[idx], y[idx], distances[idx]

    # scale distances into [0,1] for weighting
    max_d = d_nn.max()
    u = d_nn / max_d if max_d > 0 else d_nn
    weights = tricube(u)

    # fit weighted regression
    model = LinearRegression()
    model.fit(x_nn, y_nn, sample_weight=weights)

    # prediction at x0
    y_hat = model.predict(np.array([[x0]]))[0]

    # residuals for SE (simplified)
    y_pred = model.predict(x_nn)
    resid = y_nn - y_pred
    df = max(len(y_nn) - 2, 1)  # avoid div/0
    sigma2 = np.sum(weights * resid**2) / df
    se = np.sqrt(sigma2 / k)

    return y_hat, se