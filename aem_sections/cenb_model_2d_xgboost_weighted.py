import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(6)
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from aem_sections import utils
from aem_sections.utils import extract_required_aem_data, convert_to_xy, create_interp_data, create_train_test_set, plot_2d_section

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

aem_folder = '/home/sudipta/Documents/new_datasets'
log.info("reading interp data...")
all_interp_data = gpd.GeoDataFrame.from_file(
    Path(aem_folder).joinpath('interpretation_zone53_albers_study_area_Ceno_depth.shp').as_posix()
)

weight_dict = {'H': 2, 'M':1, 'L': 0.5}
all_interp_data['weight'] = all_interp_data['BoundConf'].map(weight_dict)
log.info("reading covariates ...")
original_aem_data = gpd.GeoDataFrame.from_file(Path(aem_folder).joinpath('high_res_cond_clip_albers_skip_6.shp').as_posix())


# columns
conductivities = [c for c in original_aem_data.columns if c.startswith('cond')]
d_conductivities = ['d_' + c for c in conductivities]
conduct_cols = conductivities + d_conductivities
thickness_cols = [t for t in original_aem_data.columns if t.startswith('thick')]

line_col = 'SURVEY_LIN'
lines_in_data = np.unique(all_interp_data[line_col])
train_and_val_lines_in_data, test_lines_in_data = train_test_split(lines_in_data, test_size=0.2)
train_lines_in_data, val_lines_in_data = train_test_split(train_and_val_lines_in_data, test_size=0.25)

all_lines = create_interp_data(all_interp_data, included_lines=list(lines_in_data), line_col=line_col,
                               weighted_model=True)
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(
    original_aem_data, all_lines, thickness_cols, conductivities, twod=True, include_thickness=True,
    add_conductivity_derivative=True)

if not Path('covariates_targets_2d_weights.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines, twod=True,
                         weighted_model=True)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets_2d_weights.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets_2d_weights.data', 'rb'))

train_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in train_lines_in_data]
val_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in val_lines_in_data]
test_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in test_lines_in_data]

all_data_lines = train_data_lines + val_data_lines + test_data_lines

X_train, y_train, w_train, _ = create_train_test_set(data, conduct_cols, thickness_cols, * train_data_lines)
X_val, y_val, w_val, _ = create_train_test_set(data, conduct_cols, thickness_cols, * val_data_lines)
X_test, y_test, w_test, _ = create_train_test_set(data, conduct_cols, thickness_cols, * test_data_lines)
X_train_val, y_train_val, w_train_val, _ = create_train_test_set(data, conduct_cols, thickness_cols, * train_data_lines, * val_data_lines)


def my_custom_scorer(reg, X, y):
    """learn on train data and predict on test data to ensure total out of sample validation"""
    y_val_pred = reg.predict(X_val)
    r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)
    return r2


xgb_space = {
    'max_depth': Integer(1, 15),
    'learning_rate': Real(10 ** -5, 10 ** 0, prior="log-uniform"),
    'n_estimators': Integer(20, 200),
    'min_child_weight': Integer(1, 10),
    'max_delta_step': Integer(0, 10),
    'gamma': Real(0, 0.5, prior="uniform"),
    'colsample_bytree': Real(0.3, 0.9, prior="uniform"),
    'subsample': Real(0.01, 1.0, prior='uniform'),
    'colsample_bylevel': Real(0.01, 1.0, prior='uniform'),
    'colsample_bynode': Real(0.01, 1.0, prior='uniform'),
    'reg_alpha': Real(1, 100, prior='uniform'),
    'reg_lambda': Real(0.01, 10, prior='log-uniform'),
}


def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True


# class SmoothXGBRegressor(XGBRegressor):
#
#     # def score(self, X, y, sample_weight=None):
#     #     y_pred = super().predict(X)
#     #     y_smooth = savgol_filter(y_pred, 51, 3)  # window size 51, polynomial order 3
#     #     return r2_score(y, y_smooth, sample_weight=sample_weight)
#
#     def predict(self, data, output_margin=False, ntree_limit=None, validate_features=True):
#         y_pred = super(SmoothXGBRegressor, self).predict(data, output_margin, ntree_limit, validate_features)
#         return savgol_filter(y_pred, 51, 3)  # window size 51, polynomial order 3


reg = XGBRegressor(objective='reg:squarederror', random_state=0)

searchcv = BayesSearchCV(
    reg,
    search_spaces=xgb_space,
    n_iter=48,
    cv=2,  # use 2 when using custom scoring using X_test
    verbose=1000,
    n_points=24,
    n_jobs=12,
    scoring=my_custom_scorer
)

searchcv.fit(X_train, y_train, callback=on_step)
import time
pickle.dump(searchcv, open(f"{reg.__class__.__name__}.{int(time.time())}.model", 'wb'))
# searchcv = pickle.load(open('XGBRegressionPPT2.model', 'rb'))

final_model = XGBRegressor(objective='reg:squarederror', n_jobs=3, ** searchcv.best_params_)

final_model.fit(X_train_val, y_train_val, sample_weight=w_train_val)
print(r2_score(y_train, searchcv.predict(X_train), sample_weight=w_train))
print(r2_score(y_val, searchcv.predict(X_val), sample_weight=w_val))
print(r2_score(y_test, searchcv.predict(X_test), sample_weight=w_test))
print(final_model.score(X_test, y_test, sample_weight=y_test))

# import IPython; IPython.embed(); import sys; sys.exit()

# optimised model in nci
from collections import OrderedDict

# nci_params = OrderedDict([('colsample_bylevel', 1.0),
#              ('colsample_bynode', 0.21747339007554714),
#              ('colsample_bytree', 0.3),
#              ('gamma', 0.34165675703813775),
#              ('learning_rate', 0.4416591777848587),
#              ('max_delta_step', 4),
#              ('max_depth', 9),
#              ('min_child_weight', 2),
#              ('n_estimators', 128),
#              ('reg_alpha', 8.996247710883825),
#              ('reg_lambda', 0.01),
#              ('subsample', 0.5619406651679848)])

# nci_params2 = OrderedDict([('colsample_bylevel', 1.0),
#                            ('colsample_bynode', 1.0),
#                            ('colsample_bytree', 0.3),
#                            ('gamma', 0.5),
#                            ('learning_rate', 0.011248457093432874),
#                            ('max_delta_step', 0),
#                            ('max_depth', 15),
#                            ('min_child_weight', 2),
#                            ('n_estimators', 157),
#                            ('reg_alpha', 46.93126572713199),
#                            ('reg_lambda', 0.01),
#                            ('subsample', 0.8439318640876889)])
#
# nci_params3 = OrderedDict([('colsample_bylevel', 1.0),
#                            ('colsample_bynode', 0.7938658236063535),
#                            ('colsample_bytree', 0.9),
#                            ('gamma', 0.5),
#                            ('learning_rate', 0.23487942589340755),
#                            ('max_delta_step', 9),
#                            ('max_depth', 15),
#                            ('min_child_weight', 10),
#                            ('n_estimators', 193),
#                            ('reg_alpha', 57.95496777142633),
#                            ('reg_lambda', 10.0),
#                            ('subsample', 1.0)])


plot_interp_line = test_data_lines[2]
X_val_line, y_val_line, w_val_line, X_val_line_coords = create_train_test_set(data, conduct_cols, thickness_cols, plot_interp_line)
utils.plot_2d_section(X_val_line, X_val_line_coords, plot_interp_line, final_model, 'ceno_euc_a', conductivities, thickness_cols,
                      slope=False,
                      flip_column=True, v_min=2, v_max=20)

import IPython; IPython.embed(); import sys; sys.exit()