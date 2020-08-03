import sys
import GPy
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import Optimization as op

d_ind = int(sys.argv[1])
num_run = int(sys.argv[2])
alpha_val = float(sys.argv[3])
ref_num = sys.argv[4]

def fit_gp(x, y):
    # kernel1 = GPy.kern.RBF(input_dim=x.shape[1], variance=0.1, lengthscale=0.2, ARD=True)
    # kernel2 = GPy.kern.Linear(input_dim=x.shape[1],ARD=True)
    # kernel = GPy.kern.MLP(
    #     input_dim=x.shape[1], variance=0.5, weight_variance=1.0, bias_variance=1.0
    # )
    kernel = GPy.kern.Matern52(input_dim=x.shape[1], variance=0.5, lengthscale=1.0)
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize(messages=True, max_f_eval=1000)
    return m


if d_ind == 1:
    data_path = "./Data Sets/Body Fat/"
    target = 'Bodyfat'
    data_type_dict = {'Age': 'ordinal', 'Weight': 'ordinal', 'Height': 'ordinal', 'Neck': 'ordinal', 'Chest': 'ordinal',
                      'Abdomen': 'ordinal', 'Hip': 'ordinal', 'Thigh': 'ordinal', 'Knee': 'ordinal', 'Ankle': 'ordinal',
                      'Biceps': 'ordinal', 'Forearm': 'ordinal', 'Wrist': 'ordinal', 'Bodyfat': 'ordinal',
                      'predictive_var': 'ordinal'}
    feature_set = ['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps',
                   'Forearm', 'Wrist', 'Bodyfat']
    parameters = {'tree_kind': 'regression', 'min_node_size': 1, 'max_node_depth': 5, 'prune': True, 'alpha': alpha_val,
                  'ref': True, 'data_type_dict': data_type_dict, 'response': 'Bodyfat'}
elif d_ind == 2:
    data_path = "./Data Sets/Baseball/"
    target = "Salary"
    data_type_dict = {'ba': 'ordinal', 'obp': 'ordinal', 'nrun': 'ordinal', 'nhit': 'ordinal', 'ndbl': 'ordinal',
                      'ntrpl': 'ordinal', 'nhomerun': 'ordinal', 'nrbi': 'ordinal', 'nwalk': 'ordinal',
                      'nstrk': 'ordinal', 'stolen': 'ordinal', 'nerror': 'ordinal', 'ifree': 'ordinal',
                      'ifree912': 'ordinal', 'iarb': 'ordinal', 'iarb912': 'ordinal', 'Salary': 'ordianl',
                      'predictive_var': 'ordinal'}
    feature_set = ['ba', 'obp', 'nrun', 'nhit', 'ndbl', 'ntrpl', 'nhomerun', 'nrbi', 'nwalk', 'nstrk',
                   'stolen', 'nerror', 'ifree', 'ifree912', 'iarb', 'iarb912', 'Salary']
    parameters = {'tree_kind': 'regression', 'min_node_size': 1, 'max_node_depth': 5, 'prune': True, 'alpha': alpha_val,
                  'ref': True, 'data_type_dict': data_type_dict, 'response': 'Salary'}
else:
    data_path = "./Data Sets/Auto Risk/"
    target = "symboling"
    data_type_dict = {'normalized.losses': 'ordinal', 'wheel.base': 'ordinal', 'length': 'ordinal', 'width': 'ordinal',
                      'height': 'ordinal', 'curb.weight': 'ordinal', 'engine.size': 'ordinal', 'bore': 'ordinal',
                      'stroke': 'ordinal', 'compression.ratio': 'ordinal', 'horsepower': 'ordinal', 'peak.rpm': 'ordinal',
                      'city.mpg': 'ordinal', 'highway.mpg': 'ordinal', 'price': 'ordinal', 'symboling': 'ordinal',
                      'predictive_var': 'ordinal'}
    feature_set = ['normalized.losses', 'wheel.base', 'length', 'width', 'height', 'curb.weight', 'engine.size', 'bore',
                   'stroke', 'compression.ratio', 'horsepower', 'peak.rpm', 'city.mpg', 'highway.mpg', 'price', 'symboling']
    parameters = {'tree_kind': 'regression', 'min_node_size': 1, 'max_node_depth': 5, 'prune': True, 'alpha': alpha_val,
                  'ref': True, 'data_type_dict': data_type_dict, 'response': 'symboling'}

for run in range(1, num_run+1):
    trdata = pd.read_csv(data_path + "train_" + str(run) + ".csv")
    tstdata = pd.read_csv(data_path + "test_" + str(run) + ".csv")

    nfeatures = trdata.shape[1]
    x_tr = trdata.iloc[:, 0:(nfeatures-1)]
    y_tr = trdata.iloc[:, -1].values.reshape(x_tr.shape[0], 1)

    df_tst = tstdata[feature_set]
    x_tst = tstdata.iloc[:, 0:(nfeatures-1)].values
    y_tst = tstdata.iloc[:, -1].values.reshape(x_tst.shape[0], 1)

    ''' fit the gp reference model and construct the training data for its proxy model '''
    gp_ref = fit_gp(x_tr.values, y_tr)
    gp_yhat = gp_ref.predict(x_tr.values)
    gp_yhat_mu = gp_yhat[0][:, 0]
    gp_yhat_var = gp_yhat[1][:, 0]

    x_tr[target] = gp_yhat_mu
    x_tr['predictive_var'] = gp_yhat_var

    ''' compute performance of the gp model on test data '''
    gp_pred = gp_ref.predict(x_tst)
    gp_pred_mu = gp_pred[0][:, 0]
    filename = data_path + "/result_ref_3_" + str(run) + ".txt"
    with open(filename, 'a+') as outputfile:
        outputfile.write(str(sqrt(mean_squared_error(y_tst, gp_pred_mu))) + "\n")
    outputfile.close()

    ''' fit the proxy model to the ref. model '''

    interp_util_t = op.RegressionTree()
    interp_util_t.train(tr_data=x_tr, data_type_dict=data_type_dict, parameters=parameters, output_path=data_path,
                        tst_data=df_tst, run=str(run), m_ind=ref_num)

