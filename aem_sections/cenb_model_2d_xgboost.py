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

from aem_sections.utils import extract_required_aem_data, convert_to_xy, create_interp_data, create_train_test_set, plot_2d_section

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

aem_folder = '/home/sudipta/Documents/new_datasets'
log.info("reading interp data...")
all_interp_data = gpd.GeoDataFrame.from_file(
    Path(aem_folder).joinpath('interpretation_zone53_albers_study_area_Ceno_depth.shp').as_posix()
)

log.info("reading covariates ...")
original_aem_data = gpd.GeoDataFrame.from_file(Path(aem_folder).joinpath('high_res_cond_clip_albers_skip_6.shp').as_posix())


# columns
conductivities = [c for c in original_aem_data.columns if c.startswith('cond')]
thickness = [t for t in original_aem_data.columns if t.startswith('thick')]

line_col = 'SURVEY_LIN'
lines_in_data = np.unique(all_interp_data[line_col])
train_and_val_lines_in_data, test_lines_in_data = train_test_split(lines_in_data, test_size=0.2)
train_lines_in_data, val_lines_in_data = train_test_split(train_and_val_lines_in_data, test_size=0.25)


all_lines = create_interp_data(all_interp_data, included_lines=list(lines_in_data), line_col=line_col)
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(
    original_aem_data, all_lines, thickness, conductivities, twod=True, include_thickness=True,
    add_conductivity_derivative=True)


if not Path('covariates_targets_2d.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines, twod=True)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets_2d.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets_2d.data', 'rb'))

train_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in train_lines_in_data]
val_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in val_lines_in_data]
test_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in test_lines_in_data]

all_data_lines = train_data_lines + val_data_lines + test_data_lines

X_train, y_train = create_train_test_set(data, * train_data_lines)
X_val, y_val = create_train_test_set(data, * val_data_lines)
X_test, y_test = create_train_test_set(data, * test_data_lines)


X_train_val, y_train_val = create_train_test_set(data, * train_data_lines, * val_data_lines)


def my_custom_scorer(reg, X, y):
    """learn on train data and predict on test data to ensure total out of sample validation"""
    y_val_pred = reg.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    return r2


reg = XGBRegressor(objective='reg:squarederror', random_state=0)


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


searchcv = BayesSearchCV(
    reg,
    search_spaces=xgb_space,
    n_iter=50,
    cv=2,  # use 2 when using custom scoring using X_test
    verbose=1000,
    n_points=12,
    n_jobs=3,
    scoring=my_custom_scorer
)

# searchcv.fit(X_train, y_train, callback=on_step)
import time
# pickle.dump(searchcv, open(f"{reg.__class__.__name__}.{int(time.time())}.model", 'wb'))
searchcv = pickle.load(open('XGBRegressor.1621024965.model', 'rb'))

print(r2_score(y_train, searchcv.predict(X_train)))
print(r2_score(y_val, searchcv.predict(X_val)))
print(r2_score(y_test, searchcv.predict(X_test)))


final_model = XGBRegressor(objective='reg:squarederror', n_jobs=3, ** searchcv.best_params_)
final_model.fit(X_train_val, y_train_val)

print(r2_score(y_test, final_model.predict(X_test)))
# import IPython; IPython.embed(); import sys; sys.exit()

# optimised model in nci
from collections import OrderedDict

nci_params = OrderedDict([('colsample_bylevel', 1.0),
             ('colsample_bynode', 0.21747339007554714),
             ('colsample_bytree', 0.3),
             ('gamma', 0.34165675703813775),
             ('learning_rate', 0.4416591777848587),
             ('max_delta_step', 4),
             ('max_depth', 9),
             ('min_child_weight', 2),
             ('n_estimators', 128),
             ('reg_alpha', 8.996247710883825),
             ('reg_lambda', 0.01),
             ('subsample', 0.5619406651679848)])

nci_model = XGBRegressor(objective='reg:squarederror', n_jobs=3, ** nci_params)

plot_interp_line = np.random.choice(test_data_lines)
X_val_line, y_val_line = create_train_test_set(data, plot_interp_line)
plot_2d_section(X_val_line, plot_interp_line, final_model, 'ceno_euc_a', conductivities, thickness)


import IPython; IPython.embed(); import sys; sys.exit()