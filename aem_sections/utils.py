import logging
from pathlib import Path
from typing import Tuple

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


def create_train_test_set(data, * included_interp_data):
    X = data['covariates']
    y = data['targets']
    included_lines = np.zeros(X.shape[0], dtype=bool)    # nothing is included

    for in_data in included_interp_data:
        x_max, x_min, y_max, y_min = extent_of_data(in_data)
        included_lines = included_lines | \
                           ((X.POINT_X < x_max + dis_tol) & (X.POINT_X > x_min - dis_tol) &
                            (X.POINT_Y < y_max + dis_tol) & (X.POINT_Y > y_min - dis_tol))

    return X[included_lines], y[included_lines]


def extent_of_data(data: pd.DataFrame) -> Tuple[float, float, float, float]:
    x_min, x_max = min(data['POINT_X']), max(data['POINT_X'])
    y_min, y_max = min(data['POINT_Y']), max(data['POINT_Y'])
    return x_max, x_min, y_max, y_min


def weighted_target(line_required: pd.DataFrame, tree: KDTree, x: np.ndarray):
    ind, dist = tree.query_radius(x, r=radius, return_distance=True)
    ind, dist = ind[0], dist[0]
    dist += 1e-6  # add just in case of we have a zero distance
    if len(dist):
        df = line_required.iloc[ind]
        weighted_depth = np.sum(df.Z_coor * (1 / dist) ** 2) / np.sum((1 / dist) ** 2)
        return weighted_depth
    else:
        return None


def convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, interp_data, twod=False):
    log.info("convert to xy and target values...")
    selected = []
    tree = KDTree(interp_data[twod_coords if twod else threed_coords])
    target_depths = []
    for xy, c, t in zip(aem_xy_and_other_covs.iterrows(), aem_conductivities.iterrows(), aem_thickness.iterrows()):
        i, covariates_including_xy_ = xy
        j, cc = c
        k, tt = t
        assert i == j == k
        if twod:
            x_y = covariates_including_xy_[twod_coords].values.reshape(1, -1)
            y = weighted_target(interp_data, tree, x_y)
            if y is not None:
                selected.append(covariates_including_xy_)  # in 2d conductivities are already in xy
                target_depths.append(y)
        else:
            for ccc, ttt in zip(cc, tt):
                covariates_including_xyz_ = covariates_including_xy_.append(
                    pd.Series([ttt, ccc], index=['Z_coor', 'conductivity'])
                )
                x_y_z = covariates_including_xyz_[threed_coords].values.reshape(1, -1)
                y = weighted_target(interp_data, tree, x_y_z)
                if y is not None:
                    selected.append(covariates_including_xyz_)
                    target_depths.append(y)

    X = pd.DataFrame(selected)
    y = pd.Series(target_depths, name='target', index=X.index)
    return {'covariates': X, 'targets': y}


def create_interp_data(input_interp_data, included_lines, line_col='line'):
    if not isinstance(included_lines, list):
        included_lines = [included_lines]
    line = input_interp_data[(input_interp_data['Type'] != 'WITHIN_Cenozoic')
                             & (input_interp_data['Type'] != 'BASE_Mesozoic_TOP_Paleozoic')
                             & (input_interp_data[line_col].isin(included_lines))]
    # line = add_delta(line)
    line = line.rename(columns={'DEPTH': 'Z_coor'})
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
            (line.POINT_X.iat[0]-origin[0]) ** 2 +
            (line.POINT_Y.iat[0]-origin[1]) ** 2
        )

    line['d'] = line['delta'].cumsum()
    line = line.sort_values(by=['d'], ascending=True)
    return line


from typing import List


def plot_2d_section(X_val_line: pd.DataFrame, val_interp_line: pd.DataFrame, model, col_names: List[str],
                    conductivities: List[str], thickness: List[str],
                    flip_column=False, v_min=0.3, v_max=0.8):
    if isinstance(col_names, str):
        col_names = [col_names]

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm, PowerNorm
    from matplotlib.colors import Colormap
    original_cols = X_val_line.columns[:]
    X_val_line = add_delta(X_val_line)
    origin = (X_val_line.POINT_X.iat[0], X_val_line.POINT_Y.iat[0])
    val_interp_line = add_delta(val_interp_line, origin=origin)
    d_conduct_cols = ['d_' + c for c in conductivities]
    # Z = X_val_line[conductivities]
    Z = X_val_line[d_conduct_cols]
    Z = Z - np.min(np.min((Z))) + 1.0e-10
    h = X_val_line[thickness]
    dd = X_val_line.d
    ddd = np.atleast_2d(dd).T
    d = np.repeat(ddd, h.shape[1], axis=1)
    fig, ax = plt.subplots(figsize=(40, 4))
    cmap = plt.get_cmap('viridis')

    # Normalize(vmin=0.3, vmax=0.6) d(cond) norm
    im = ax.pcolormesh(d, -h, Z, norm=LogNorm(vmin=0.2, vmax=0.5), cmap=cmap, linewidth=1, rasterized=True)
    fig.colorbar(im, ax=ax)
    axs = ax.twinx()
    ax.plot(X_val_line.d, -model.predict(X_val_line[original_cols]), label='prediction', linewidth=2, color='r')
    ax.plot(val_interp_line.d, -val_interp_line.Z_coor, label='interpretation', linewidth=2, color='k')
    # for c in col_names:
    #     axs.plot(X_val_line.d, -X_val_line[c] if flip_column else X_val_line[c], label=c, linewidth=2, color='orange')

    ax.set_xlabel('distance along aem line (m)')
    ax.set_ylabel('depth (m)')
    plt.title("d(Conductivity) vs depth")

    ax.legend()
    axs.legend()
    plt.show()