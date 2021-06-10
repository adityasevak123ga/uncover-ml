import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import KDTree

log = logging.getLogger(__name__)

# distance within which an interpretation point is considered to contribute to target values
radius = 500
cell_size = 10
dis_tol = 100  # meters, distance tolerance used
twod_coords = ['POINT_X', 'POINT_Y']
threed_coords = twod_coords + ['Z_coor']
aem_covariate_cols = ['ceno_euc_a', 'Gravity_la', 'national_W', 'relief_ele', 'relief_mrv', 'SagaWET9ce'] \
                     + ['elevation', 'tx_height']


# categorical = 'relief_mrv'
# covariate_cols_without_xyz = aem_covariate_cols + ['conductivity']
# final_cols = coords + aem_covariate_cols + ['Z_coor']


def extract_required_aem_data(in_scope_aem_data, interp_data, thickness, conductivities, twod=False,
                              include_thickness=False, add_conductivity_derivative=False):
    # find bounding box
    x_max, x_min, y_max, y_min = extent_of_data(interp_data)
    # use bbox to select data only for one line
    aem_data = in_scope_aem_data[
        (in_scope_aem_data.POINT_X < x_max + dis_tol) &
        (in_scope_aem_data.POINT_X > x_min - dis_tol) &
        (in_scope_aem_data.POINT_Y < y_max + dis_tol) &
        (in_scope_aem_data.POINT_Y > y_min - dis_tol)
        ]
    aem_data = aem_data.sort_values(by='POINT_Y', ascending=False)
    aem_data[thickness] = aem_data[thickness].cumsum(axis=1)
    conduct_cols = conductivities[:] if twod else []
    thickness_cols = thickness if include_thickness else []
    if twod and add_conductivity_derivative:
        conductivity_diff = aem_data[conduct_cols].diff(axis=1, periods=-1)
        conductivity_diff.fillna(axis=1, method='ffill', inplace=True)
        d_conduct_cols = ['d_' + c for c in conduct_cols]
        aem_data[d_conduct_cols] = conductivity_diff
        conduct_cols += d_conduct_cols

    aem_xy_and_other_covs = aem_data[twod_coords + aem_covariate_cols + conduct_cols + thickness_cols]
    aem_conductivities = aem_data[conductivities]
    aem_thickness = aem_data[thickness]
    return aem_xy_and_other_covs, aem_conductivities, aem_thickness


def create_train_test_set(data, conduct_cols, thickness, *included_interp_data,
                          included_cols=Optional[List],
                          weighted_model=False):
    X = data['covariates']
    y = data['targets']
    w = data['weight']
    included_lines = np.zeros(X.shape[0], dtype=bool)  # nothing is included

    for in_data in included_interp_data:
        x_max, x_min, y_max, y_min = extent_of_data(in_data)
        included_lines = included_lines | \
                         ((X.POINT_X < x_max + dis_tol) & (X.POINT_X > x_min - dis_tol) &
                          (X.POINT_Y < y_max + dis_tol) & (X.POINT_Y > y_min - dis_tol))

    if included_cols:
        cols = included_cols
    else:  # include all covairates + cols
        cols = conduct_cols + thickness + aem_covariate_cols

    return X[included_lines][cols], y[included_lines], w[included_lines], X[included_lines][twod_coords]


def extent_of_data(data: pd.DataFrame) -> Tuple[float, float, float, float]:
    x_min, x_max = min(data['POINT_X']), max(data['POINT_X'])
    y_min, y_max = min(data['POINT_Y']), max(data['POINT_Y'])
    return x_max, x_min, y_max, y_min


def weighted_target(line_required: pd.DataFrame, tree: KDTree, x: np.ndarray, weighted_model):
    ind, dist = tree.query_radius(x, r=radius, return_distance=True)
    ind, dist = ind[0], dist[0]
    if len(dist):
        dist += 1e-6  # add just in case of we have a zero distance
        df = line_required.iloc[ind]
        weighted_depth = np.sum(df.Z_coor * (1 / dist) ** 2) / np.sum((1 / dist) ** 2)
        if weighted_model:
            weighted_weight = np.sum(df.weight * (1 / dist) ** 2) / np.sum((1 / dist) ** 2)
        else:
            weighted_weight = None

        return weighted_depth, weighted_weight
    else:
        return None, None


def convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, interp_data, twod=False,
                  weighted_model=False):
    log.info("convert to xy and target values...")
    selected = []
    tree = KDTree(interp_data[twod_coords if twod else threed_coords])
    target_depths = []
    target_weights = []
    for xy, c, t in zip(aem_xy_and_other_covs.iterrows(), aem_conductivities.iterrows(), aem_thickness.iterrows()):
        i, covariates_including_xy_ = xy
        j, cc = c
        k, tt = t
        assert i == j == k
        if twod:
            x_y = covariates_including_xy_[twod_coords].values.reshape(1, -1)
            y, w = weighted_target(interp_data, tree, x_y, weighted_model)
            if y is not None:
                if weighted_model:
                    if w is not None:
                        selected.append(covariates_including_xy_)  # in 2d conductivities are already in xy
                        target_depths.append(y)
                        target_weights.append(w)
                else:
                    selected.append(covariates_including_xy_)  # in 2d conductivities are already in xy
                    target_depths.append(y)
                    target_weights.append(1.0)
        else:
            for ccc, ttt in zip(cc, tt):
                covariates_including_xyz_ = covariates_including_xy_.append(
                    pd.Series([ttt, ccc], index=['Z_coor', 'conductivity'])
                )
                x_y_z = covariates_including_xyz_[threed_coords].values.reshape(1, -1)
                y, w = weighted_target(interp_data, tree, x_y_z, weighted_model)
                if y is not None:
                    selected.append(covariates_including_xyz_)
                    target_depths.append(y)

    X = pd.DataFrame(selected)
    y = pd.Series(target_depths, name='target', index=X.index)
    w = pd.Series(target_weights, name='weight', index=X.index)
    return {'covariates': X, 'targets': y, 'weights': w}


def create_interp_data(input_interp_data, included_lines, line_col='line', weighted_model=False):
    if not isinstance(included_lines, list):
        included_lines = [included_lines]
    line = input_interp_data[(input_interp_data['Type'] != 'WITHIN_Cenozoic')
                             & (input_interp_data['Type'] != 'BASE_Mesozoic_TOP_Paleozoic')
                             & (input_interp_data[line_col].isin(included_lines))]
    # line = add_delta(line)
    line = line.rename(columns={'DEPTH': 'Z_coor'})
    if weighted_model:
        line_required = line[threed_coords + ['weight']]
    else:
        line_required = line[threed_coords]
    return line_required


def add_delta(line, origin=None):
    line = line.sort_values(by='POINT_Y', ascending=False)
    line['POINT_X_diff'] = line['POINT_X'].diff()
    line['POINT_Y_diff'] = line['POINT_Y'].diff()
    line['delta'] = np.sqrt(line.POINT_X_diff ** 2 + line.POINT_Y_diff ** 2)
    line['delta'] = line['delta'].fillna(value=0.0)
    if origin is not None:
        line['delta'].iat[0] = np.sqrt(
            (line.POINT_X.iat[0] - origin[0]) ** 2 +
            (line.POINT_Y.iat[0] - origin[1]) ** 2
        )

    line['d'] = line['delta'].cumsum()
    line = line.sort_values(by=['d'], ascending=True)
    return line


from typing import List


def plot_2d_section(X_val_line: pd.DataFrame,
                    X_val_line_coords: pd.DataFrame,
                    val_interp_line: pd.DataFrame, model, col_names: List[str],
                    conductivities: List[str], thickness: List[str], slope=False,
                    flip_column=False, v_min=0.3, v_max=0.8):
    if isinstance(col_names, str):
        col_names = [col_names]

    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm, PowerNorm
    from matplotlib.colors import Colormap
    original_cols = X_val_line.columns[:]
    line = add_delta(X_val_line_coords)
    X_val_line = pd.concat([X_val_line, line], axis=1)
    origin = (X_val_line.POINT_X.iat[0], X_val_line.POINT_Y.iat[0])
    val_interp_line = add_delta(val_interp_line, origin=origin)
    if slope:
        d_conduct_cols = ['d_' + c for c in conductivities]
        Z = X_val_line[d_conduct_cols]
        Z = Z - np.min(np.min((Z))) + 1.0e-10
    else:
        Z = X_val_line[conductivities]

    h = X_val_line[thickness]
    dd = X_val_line.d
    ddd = np.atleast_2d(dd).T
    d = np.repeat(ddd, h.shape[1], axis=1)
    fig, ax = plt.subplots(figsize=(40, 4))
    cmap = plt.get_cmap('viridis')

    if slope:
        norm = LogNorm(vmin=v_min, vmax=v_max)
    else:
        norm = Normalize(vmin=v_min, vmax=v_max)

    im = ax.pcolormesh(d, -h, Z, norm=norm, cmap=cmap, linewidth=1, rasterized=True)
    fig.colorbar(im, ax=ax)
    axs = ax.twinx()
    y_pred = -model.predict(X_val_line[original_cols])
    pred = savgol_filter(y_pred, 11, 3)  # window size 51, polynomial order 3
    ax.plot(X_val_line.d, pred, label='prediction', linewidth=2, color='r')
    ax.plot(val_interp_line.weight_dict, -val_interp_line.Z_coor, label='interpretation', linewidth=2, color='k')
    # for c in col_names:
    #     axs.plot(X_val_line.d, -X_val_line[c] if flip_column else X_val_line[c], label=c, linewidth=2, color='orange')

    ax.set_xlabel('distance along aem line (m)')
    ax.set_ylabel('depth (m)')
    if slope:
        plt.title("d(Conductivity) vs depth")
    else:
        plt.title("Conductivity vs depth")

    ax.legend()
    axs.legend()
    plt.show()
