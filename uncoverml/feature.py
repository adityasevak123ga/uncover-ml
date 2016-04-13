import os.path
import numpy as np
import tables as hdf

from uncoverml import geoio
from uncoverml import patch


def output_features(feature_vector, outfile, featname="features"):
    """
    Writes a vector of features out to a standard HDF5 format. The function
    assumes that it is only 1 chunk of a larger vector, so outputs a numerical
    suffix to the file as an index.

    Parameters
    ----------
        feature_vector: array
            A 2D numpy array of shape (nPoints, nDims) of type float. This can
            be a masked array.
        outfile: path
            The name of the output file
        featname: str, optional
            The name of the features.
    """
    h5file = hdf.open_file(outfile, mode='w')
    array_shape = feature_vector.shape

    filters = hdf.Filters(complevel=5, complib='zlib')

    if np.ma.isMaskedArray(feature_vector):
        fobj = feature_vector.data
        fmask = feature_vector.mask
    else:
        fobj = feature_vector
        fmask = np.zeros(array_shape, dtype=bool)

    h5file.create_carray("/", featname, filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape, obj=fobj)
    h5file.create_carray("/", "mask", filters=filters,
                         atom=hdf.BoolAtom(), shape=array_shape, obj=fmask)
    h5file.close()

def patches_from_image(image, patchsize, targets=None):
    """
    Pulls out masked patches from a geotiff, either everywhere or 
    at locations specificed by a targets shapefile
    """
    # Get the target points if they exist:
    data_and_mask = image.data()
    data = data_and_mask.data
    data_dtype = data.dtype
    mask = data_and_mask.mask
    pixels = None
    if targets is not None:
        lonlats = geoio.points_from_hdf(targets)
        inx = np.logical_and(lonlats[:, 0] >= image.xmin,
                             lonlats[:, 0] < image.xmax)
        iny = np.logical_and(lonlats[:, 1] >= image.ymin,
                             lonlats[:, 1] < image.ymax)
        valid = np.logical_and(inx, iny)
        valid_lonlats = lonlats[valid]
        pixels = image.lonlat2pix(valid_lonlats, centres=True)
        patches = patch.point_patches(data, patchsize, pixels)
        patch_mask = patch.point_patches(mask, patchsize, pixels)
    else:
        patches = patch.grid_patches(data, patchsize)
        patch_mask = patch.grid_patches(mask, patchsize)

    patch_data = np.array(list(patches), dtype=data_dtype)
    mask_data = np.array(list(patch_mask), dtype=bool)
    result = np.ma.masked_array(data=patch_data, mask=mask_data)
    return result

def __load_hdf5(infiles):
    for filename in infiles:
        data_list = []
        with hdf.open_file(filename, mode='r') as f:
            data = f.root.features[:]
            mask = f.root.mask[:]
            a = np.ma.masked_array(data=data, mask=mask)
            data_list.append(a)
        all_data = np.concatenate(data_list, axis=1)
        return all_data

def load_features(filename_dict, chunk_indices):
    """
    we load references to the data into each node, this function runs
    on the node to actually load the data itself.
    """
    multi_hdf5_type = [type(k) is list for k in reference_dict.values()]
    is_hdf5 = np.all(np.array(multi_hdf5_type))
    data_dict = {}
    for i in chunk_indices:
        inobj = reference_dict[i]
        data_dict[i] = __load_hdf5(inobj) if is_hdf5 else __load_image(inobj) 
    return data_dict

def load_image_data(image_dict, chunk_indices, patchsize, targets):
    """
    we load references to the data into each node, this function runs
    on the node to actually load the data itself.
    """
    data_dict = {i:patches_from_image(image_dict[i], patchsize, targets)
        for i in chunk_indices}
    return data_dict

def image_data_vector(image_data):
    """
    image_data : dictionary of ndarrays
    """
    indices = sorted(image_data.keys())
    data_list = []
    for i in indices:
        d = image_data[i]
        data_list.append(d.reshape((d.shape[0],-1)))
    x = np.ma.concatenate(data_list, axis=0)
    return x

def data_vector(feature_dict):
    indices = sorted(data_dict.keys())
    in_d = []
    for i in indices:
        if data_dict[i].ndim == 3:
            x = data_dict[i].reshape((-1, data_dict[i].shape[2]))
            in_d.append(x)
        else:
            in_d.append(data_dict[i])
    x = np.ma.concatenate(in_d, axis=0)
    return x


def cat_chunks(filename_chunks):

    feats = []
    masks = []
    for i, flist in filename_chunks.items():
        feat = []
        mask = []
        for f in flist:
            f, m = input_features(f)
            feat.append(f)
            mask.append(m)
        feats.append(np.hstack(feat))
        masks.append(np.hstack(mask))

    X = np.vstack(feats)
    M = np.vstack(masks)
    return X, M
