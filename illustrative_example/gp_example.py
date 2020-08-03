import numpy as np
import pandas as pd
import GPy
import scipy.special
from matplotlib import pyplot as plt
import sys

sys.path.append("..")
import Optimization as op

def generate_data_invlogit(n, a, b, s):
    x = np.linspace(-1, 1, n)
    y_noiseless = scipy.special.expit(a * x + b)
    y = y_noiseless + s * np.random.randn(n)

    return x, y, y_noiseless


def generate_data_gp(n, w, v, b, s):
    x = np.linspace(-1, 1, n)
    kernel = GPy.kern.MLP(input_dim=1, variance=v, weight_variance=w, bias_variance=b)
    y_noiseless = np.linalg.cholesky(
        kernel.K(x[:, np.newaxis]) + 1e-6 * np.eye(n)
    ) @ np.random.randn(n)
    y = y_noiseless + s * np.random.randn(n)

    return x, y, y_noiseless


def fit_gp(x, y):
    kernel = GPy.kern.MLP(
        input_dim=1, variance=1.0, weight_variance=1.0, bias_variance=1.0
    )
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize_restarts(num_restarts=10)

    return m


if __name__ == "__main__":
    np.random.seed(123)

    n = 150
    n_plot = 1001
    sigma = 1.0
    tree_depths = [1, 2, 3, 4, 5, 6]
    plot_i = 3  # plot mean functions of trees for models at depth tree_depths[plot_i]

    # generate data
    w, v, b = 5.0, 5.0, 1.0
    x_f, y_f_noisy, y_f = generate_data_gp(n_plot, w, v, b, sigma)
    x = x_f[0 : n_plot : int(n_plot / n)]
    y = y_f_noisy[0 : n_plot : int(n_plot / n)]

    # fit GP as reference model
    gp = fit_gp(x[:, np.newaxis], y[:, np.newaxis])
    gp_yhat = gp.predict(x[:, np.newaxis])
    gp_yhat_mu = gp_yhat[0][:, 0]
    gp_yhat_var = gp_yhat[1][:, 0]
    gp_yhat_f = gp.predict(x_f[:, np.newaxis])[0][:, 0]

    # put data into data frames
    dat_f = pd.DataFrame(data={"x": x_f})
    dat_types = {"x": "ordinal", "y": "ordinal"}
    dat = pd.DataFrame(data={"x": x, "y": y})
    dat_ref_types = {"x": "ordinal", "y": "ordinal", "predictive_var": "ordinal"}
    dat_ref = pd.DataFrame(
        data={"x": x, "y": gp_yhat_mu, "predictive_var": gp_yhat_var}
    )

    # regression tree parameters
    parameters = {
        "tree_kind": "regression",
        "min_node_size": 1,
        "max_node_depth": None,
        "alpha": 0,
        "prune": False,
        "ref": False,
        "data_type_dict": dat_types,
        "response": "y",
    }

    yhat = np.zeros((len(tree_depths), 2, n_plot))
    for i, depth in enumerate(tree_depths):
        parameters["max_node_depth"] = depth

        # fit to training data
        parameters["proj"] = False
        rt_data = op.RegressionTree()
        rt_data.train(tr_data=dat, data_type_dict=dat_types, parameters=parameters)
        y_f_hat_rt_data = rt_data.predict(dat_f)
        yhat[i, 0, :] = y_f_hat_rt_data

        # fit to reference model
        parameters["proj"] = True
        rt_ref = op.RegressionTree()
        rt_ref.train(tr_data=dat_ref, data_type_dict=dat_ref_types, parameters=parameters)
        y_f_hat_rt_ref = rt_ref.predict(dat_f)
        yhat[i, 1, :] = y_f_hat_rt_ref

    # plot
    plt.figure(figsize=(8, 8))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=20)
    plt.xlabel('xlabel', fontsize=20)
    plt.ylabel('ylabel', fontsize=20)
    plt.plot(x_f, y_f, "k-", label="true function", linewidth=2.5)
    plt.plot(x, y, "r.", label="training data")
    plt.plot(x_f, gp_yhat_f, "-", label="ref.model", color="green", linewidth=2.5)
    plt.plot(x_f, yhat[plot_i, 0, :], "b-", label="tree fit to training data", linewidth=2.5)
    plt.plot(x_f, yhat[plot_i, 1, :], "m-", label="tree fit to ref.model", linewidth=2.5)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()
    plt.savefig('illustrative_example.pdf',dpi=400)
    plt.show()

    rmse_ref = np.sqrt(np.mean((gp_yhat_f - y_f) ** 2))
    rmse = np.sqrt(np.mean((yhat - y_f) ** 2, axis=2))
    plt.figure(figsize=(8, 8))
    plt.plot(tree_depths, rmse_ref * np.ones(len(tree_depths)), "-", label="ref.model", color="green", linewidth=2.5)
    plt.plot(tree_depths, rmse[:, 0], "b-", label="tree fit to training data", linewidth=2.5)
    plt.plot(tree_depths, rmse[:, 1], "m-", label="tree fit to ref.model", linewidth=2.5)
    plt.ylabel("RMSE", fontsize=20)
    plt.xlabel("Tree depth", fontsize=20)
    plt.legend()
    plt.savefig('rmse_illustrative_example.pdf', dpi=400)
    plt.show()
