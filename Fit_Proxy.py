import sys
import pandas as pd
import Optimization as op

run = sys.argv[1]
d_ind = int(sys.argv[2])
alpha_val = float(sys.argv[3])
ref_num = sys.argv[4]

if d_ind == 1:
    data_path = "./Data Sets/Body Fat"
    data_type_dict = {'Age': 'ordinal', 'Weight': 'ordinal', 'Height': 'ordinal', 'Neck': 'ordinal', 'Chest': 'ordinal',
                      'Abdomen': 'ordinal', 'Hip': 'ordinal', 'Thigh': 'ordinal', 'Knee': 'ordinal', 'Ankle': 'ordinal',
                       'Biceps': 'ordinal', 'Forearm': 'ordinal', 'Wrist': 'ordinal', 'Bodyfat': 'ordinal'}
    feature_set = ['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps',
                   'Forearm', 'Wrist', 'Bodyfat']
    parameters = {'tree_kind': 'regression', 'min_node_size': 1, 'max_node_depth': 5, 'prune': True, 'alpha': alpha_val,
                  'ref': False, 'data_type_dict': data_type_dict, 'response': 'Bodyfat'}
elif d_ind == 2:
    data_path = "./Data Sets/Baseball"
    data_type_dict = {'Salary': 'ordianl', 'ba': 'ordinal', 'obp': 'ordinal', 'nrun': 'ordinal', 'nhit': 'ordinal',
                      'ndbl': 'ordinal', 'ntrpl': 'ordinal', 'nhomerun': 'ordinal', 'nrbi': 'ordinal',
                      'nwalk': 'ordinal', 'nstrk': 'ordinal', 'stolen': 'ordinal', 'nerror': 'ordinal',
                      'ifree': 'ordinal', 'ifree912': 'ordinal', 'iarb': 'ordinal', 'iarb912': 'ordinal'}
    feature_set = ['Salary', 'ba', 'obp', 'nrun', 'nhit', 'ndbl', 'ntrpl', 'nhomerun', 'nrbi', 'nwalk', 'nstrk',
                   'stolen', 'nerror', 'ifree', 'ifree912', 'iarb', 'iarb912']
    parameters = {'tree_kind': 'regression', 'min_node_size': 1, 'max_node_depth': 5, 'prune': True, 'alpha': alpha_val,
                  'ref': False, 'data_type_dict': data_type_dict, 'response': 'Salary'}
else:
    data_path = "./Data Sets/Auto Risk"
    data_type_dict = {'normalized.losses': 'ordinal', 'wheel.base': 'ordinal', 'length': 'ordinal', 'width': 'ordinal',
                      'height': 'ordinal', 'curb.weight': 'ordinal', 'engine.size': 'ordinal', 'bore': 'ordinal',
                      'stroke': 'ordinal', 'compression.ratio': 'ordinal', 'horsepower': 'ordinal', 'peak.rpm': 'ordinal',
                      'city.mpg': 'ordinal', 'highway.mpg': 'ordinal', 'price': 'ordinal', 'symboling': 'ordinal'}
    feature_set = ['normalized.losses', 'wheel.base', 'length', 'width', 'height', 'curb.weight', 'engine.size', 'bore',
                   'stroke', 'compression.ratio', 'horsepower', 'peak.rpm', 'city.mpg', 'highway.mpg', 'price', 'symboling']
    parameters = {'tree_kind': 'regression', 'min_node_size': 1, 'max_node_depth': 5, 'prune': True, 'alpha': alpha_val,
                  'ref': False, 'data_type_dict': data_type_dict, 'response': 'symboling'}

''' fitt the proxy model to the training data '''

trainpath = data_path + "/train_" + str(run) + ".csv"
df_tr = pd.read_csv(trainpath)
df_tr = df_tr[feature_set]

testpath = data_path + "/test_" + str(run) + ".csv"
df_tst = pd.read_csv(testpath)
df_tst = df_tst[feature_set]

interp_prior_t = op.RegressionTree()
interp_prior_t.train(tr_data=df_tr, data_type_dict=data_type_dict, parameters=parameters, output_path=data_path,
                     tst_data=df_tst, run=run, m_ind=0)

''' fit the proxy model to the ref. model '''

trainpath = data_path + "/proxy_train_" + str(ref_num) + "_" + str(run) + ".csv"
df_tr = pd.read_csv(trainpath)
feature_set.append('predictive_var')
df_tr = df_tr[feature_set]
data_type_dict['predictive_var'] = 'ordinal'

parameters['ref'] = True
interp_util_t = op.RegressionTree()
interp_util_t.train(tr_data=df_tr, data_type_dict=data_type_dict, parameters=parameters, output_path=data_path,
             tst_data=df_tst, run=str(run), m_ind=ref_num)

