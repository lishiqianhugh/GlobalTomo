import numpy as np
from pathlib import Path
import h5py
from modulus.sym.dataset import Dataset
from typing import Dict, List, Union


def load_wf(h5f, model_ids, snapshot_ids, pred_key, data_type, source=None, test=False):
    """
    Load waveform data from a specified HDF5 file and prepare it according to the given parameters for further analysis.

    Parameters:
        h5f (h5py.File): An open HDF5 file object containing wavefield data and other related attributes.
        model_ids (list of int): List of model indices to specify which models' data to load from the HDF5 file.
        snapshot_ids (list of int): Indices to specify which time snapshots to load from the waveform data.
        pred_key (str): Key to select the specific type of waveform data from the HDF5 file.
        data_type (str): Specifies the format for organizing the output data. Options include:
            - 'point': Data is reshaped to a point format, suitable for point-based analyses.
            - 'vector': Data is formatted as vectors where spatial and temporal points are organized in one vector.
            - 'grid': Data is organized into a grid structure, ideal for grid-based analyses like image processing.
        source (.h5): An open HDF5 file object containing source data and related attributes. Defaults to None.
        test (bool, optional): If True, additional testing-specific data are loaded and processed, including scaled Cartesian coordinates and specific time snapshots. Defaults to False.

    Returns:
        tuple of (dict, dict):
            - The first dictionary ('invar') contains the input variables such as coordinates, time, and harmonics, organized according to the specified 'data_type'.
            - The second dictionary ('outvar') contains the output wavefield data formatted according to 'data_type' and indexed by 'pred_key'.

    Note:
        The function assumes a consistent structure and presence of necessary keys within the HDF5 file, such as 'element_coords_cartesian', 'harmonics', and 'X'. 
    """

    model_ids.sort()
    invar = {}
    outvar = {}
    coords = h5f["element_coords_cartesian"][:,:,:]
    if test:
        invar['x'] = coords[None, :, :, 0].repeat(len(model_ids), 0).reshape(len(model_ids), -1)  / 1e3 # b, 16*3648
        invar['y'] = coords[None, :, :, 1].repeat(len(model_ids), 0).reshape(len(model_ids), -1)  / 1e3 # b, 16*3648
        invar['z'] = coords[None, :, :, 2].repeat(len(model_ids), 0).reshape(len(model_ids), -1)  / 1e3 # b, 16*3648
        time = np.array([0.0744022, 0.2740638, 0.4737254, 0.673387, 0.8730486, 1.0727102, 1.2723718, 
                    1.4720334, 1.671695, 1.8713566, 2.0710182, 2.2706798, 2.4703414, 2.670003, 2.8696646])
        invar['t'] = time[snapshot_ids][None, :].repeat(len(model_ids), axis=0)
 
    invar['h'] = h5f['harmonics'][model_ids]
    if source is not None:
        s = source['source'][model_ids] / 1e10 / 4e1
        invar['h'] = np.concatenate([invar['h'], s], axis=-1)
    
    if 'X' in h5f.keys():
        outvar[pred_key] = h5f["X"][model_ids][:,snapshot_ids] * 1e5
    else:
        outvar[pred_key] = h5f["disp"][model_ids][:,snapshot_ids,:,:,2:] * 1e10
        
    if data_type == 'vector':
        for key in invar.keys():
            invar[key] = invar[key].reshape(len(model_ids), -1)
        for key in outvar.keys():
            outvar[key] = outvar[key].reshape(len(model_ids), -1)
    elif data_type == 'point':
        invar['x'] = coords[None, :, :, 0].repeat(len(model_ids), 0).reshape(len(model_ids), -1)  / 1e3 # b, 16*3648
        invar['y'] = coords[None, :, :, 1].repeat(len(model_ids), 0).reshape(len(model_ids), -1)  / 1e3 # b, 16*3648
        invar['z'] = coords[None, :, :, 2].repeat(len(model_ids), 0).reshape(len(model_ids), -1)  / 1e3 # b, 16*3648
        time = np.array([0.0744022, 0.2740638, 0.4737254, 0.673387, 0.8730486, 1.0727102, 1.2723718, 
                    1.4720334, 1.671695, 1.8713566, 2.0710182, 2.2706798, 2.4703414, 2.670003, 2.8696646])
        invar['t'] = time[snapshot_ids][None, :].repeat(len(model_ids), axis=0) # b, 4
        for key in invar.keys():
            if key in ['h']:
                invar[key] = invar[key].reshape(len(model_ids), -1)
        for key in outvar.keys():
            outvar[key] = outvar[key].reshape(len(model_ids), -1, 1) # b, 4*16*3648, 1
    else:
        raise ValueError("data_type should be one of ['point', 'vector']")
    
    return invar, outvar

def load_seis(h5f, model_ids, pred_key, data_type, norm=False, baseline=None, source=None, time_range=[0,150], transpose=True, test=False):
    """
    Load seismic data from a given HDF5 file and prepare it for analysis based on specified parameters.

    Parameters:
        h5f (h5py.File): An open HDF5 file object containing seismic data and related attributes.
        model_ids (list of int): Indices to specify which models from the HDF5 file to load data for.
        pred_key (str): Key to select the specific type of displacement data ('disp_x', 'disp_y', 'disp_z') from the HDF5 file.
        data_type (str): Specifies how the output data should be formatted. Options include:
            - 'point': Data is reshaped to represent points, ideal for point-based processing.
            - 'vector': Data is prepared in vector form where spatial and temporal points are organized in one vector.
            - 'grid': Data is reshaped into a grid format, typically used for image or grid-based analyses.
        norm (bool, optional): If True, normalize the output displacement data. Defaults to False.
        baseline (float, optional): A baseline value to be subtracted from the displacement data. Defaults to None.
        source (.h5, optional): An open HDF5 file object containing source data and related attributes. Defaults to None.
        time_range (list of int, optional): Indices to specify the time range of the data to load. Defaults to [0, 150].    
        transpose (bool, optional): If True, transpose the output data to match the dimensions (batch, time, width, height). Defaults to True.
        test (bool, optional): If True, additional testing-specific data are loaded and processed. Defaults to False.

    Returns:
        tuple of (dict, dict):
            - The first dictionary ('invar') contains input variables such as time, coordinates, and harmonics, formatted according to the 'data_type'.
            - The second dictionary ('outvar') contains output variables like displacement, adjusted according to normalization and baseline settings.

    Note:
        The function assumes the structure and keys of the HDF5 file are consistent with expected schemas, like having keys for 'time', 'disp', and coordinate data.
    """

    # Initialize and sort model IDs
    model_ids.sort()
    invar = {}
    outvar = {}

    # Load basic data
    sph_coords = h5f["station_coords_spherical"][model_ids]
    time = h5f["time"][model_ids][:, time_range[0]:time_range[1]]
    disp = h5f["disp"][model_ids][:,:,:,:,time_range[0]:time_range[1]]

    # Handle test-specific data
    if test:
        car_coords = h5f["station_coords_cartesian"][model_ids]
        invar.update({
            "x": car_coords[:,:,:,0] / 1e3,  # (b, 37, 37)
            "y": car_coords[:,:,:,1] / 1e3,  # (b, 37, 37)
            "z": car_coords[:,:,:,2] / 1e3,  # (b, 37, 37)
            "xs": sph_coords[:,:,:,0],  # r
            "ys": sph_coords[:,:,:,1] / np.pi,  # theta
            "zs": sph_coords[:,:,:,2] / np.pi,  # phi
            "t": time
        })

    # Prepare output variables based on prediction key
    key2idx = {'disp_x': 0, 'disp_y': 1, 'disp_z': 2}
    outvar[pred_key] = disp[:,:,:,key2idx[pred_key],:] * 1e10

    # Adjust baseline if necessary
    if baseline is not None:
        outvar[pred_key] -= baseline * 1e10

    # Normalize if requested
    if norm:
        max_value = np.max(np.abs(outvar[pred_key]), axis=-1)
        outvar[pred_key] /= max_value[:, :, :, None]

    # Transpose output to match dimensions (batch, time, width, height)
    if transpose:
        outvar[pred_key] = outvar[pred_key].transpose(0, 3, 1, 2)
    
    invar['h'] = h5f['harmonics'][model_ids]
    if source is not None:
        s = source['source'][model_ids] / 1e10 / 4e1
        invar['h'] = np.concatenate([invar['h'], s], axis=-1)

    # Handle different data types
    if data_type == 'vector':
        t_size, w_size, h_size = time_range[1] - time_range[0], 37, 37
        for key in invar:
            if key not in ['t', 'h']:
                invar[key] = np.repeat(invar[key][:,np.newaxis,:,:], t_size, axis=1)
        if 't' in invar:
            invar['t'] = np.repeat(time[:, :, np.newaxis], w_size * h_size, axis=-1).reshape(len(model_ids), t_size, w_size, h_size)
        for key in invar:
            invar[key] = invar[key].reshape(len(model_ids), -1)
        for key in outvar:
            outvar[key] = outvar[key].reshape(len(model_ids), -1)
    elif data_type == 'point':
        car_coords = h5f["station_coords_cartesian"][model_ids]
        invar.update({
            "x": car_coords[:,:,:,0] / 1e3,
            "y": car_coords[:,:,:,1] / 1e3,
            "z": car_coords[:,:,:,2] / 1e3,
            "t": time
        })
        for key in invar:
            if key not in ['t', 'h']:
                invar[key] = invar[key].reshape(len(model_ids), -1)
        for key in outvar:
            outvar[key] = outvar[key].reshape(len(model_ids), -1, 1)
    else:
        raise ValueError("data_type should be one of ['point', 'vector']")

    return invar, outvar

def load_interior(h5f, model_ids, snapshot_ids):
    # invar: h x z t m 
    model_ids.sort()
    invar = {}
    outvar = {}
    timestep = 10
    invar["h"] = h5f["h"][model_ids] # b, 405
    for key in ['x', 'y', 'z']:
        invar[key] = h5f[key][:].reshape(181,361,5)[:,180,1:].reshape(-1)
        invar[key] = invar[key][None, :].repeat(len(model_ids), 0)  # b, 181*361*5
    
    # invar["t"] = h5f["t"][:][None, 10::timestep].repeat(len(model_ids), 0)  # b, 150 / timestep
    time = np.array([0.0744022, 0.2740638, 0.4737254, 0.673387, 0.8730486, 1.0727102, 1.2723718, 
            1.4720334, 1.671695, 1.8713566, 2.0710182, 2.2706798, 2.4703414, 2.670003, 2.8696646])
    invar["t"] = time[snapshot_ids][None, :].repeat(len(model_ids), 0)  # b, 150 / timestep
    invar["vp"] = h5f["m"][model_ids].reshape(len(model_ids), 181, 361, 5)[:, :, 180, 1:] + 1 # b, 181*361*5
    invar["vp"] = invar["vp"].reshape(len(model_ids), -1)
    invar["vs"] = np.zeros_like(invar["vp"])
    # outvar["wave_equation"] = np.zeros((len(model_ids), invar["x"].shape[-1]*invar["t"].shape[-1], invar["theta"].shape[-1]))
    outvar["wave_equation"] = np.zeros((len(model_ids), invar["x"].shape[-1]*invar["t"].shape[-1]))

    return invar, outvar

def load_surface(h5f, model_ids, snapshot_ids):
    invar = {}
    outvar = {}
    invar["h"] = h5f["h"][model_ids] # b, 405
    for key in ['x', 'y', 'z']:
        invar[key] = h5f[key][:].reshape(181,361)[:,180].reshape(-1)
        invar[key] = invar[key][None, :].repeat(len(model_ids), 0)  # b, 181*361*5
    time = np.array([0.0744022, 0.2740638, 0.4737254, 0.673387, 0.8730486, 1.0727102, 1.2723718, 
            1.4720334, 1.671695, 1.8713566, 2.0710182, 2.2706798, 2.4703414, 2.670003, 2.8696646])
    invar["t"] = time[snapshot_ids][None, :].repeat(len(model_ids), 0)  # b, 150 / timestep
    outvar["free_surface"] = np.zeros((len(model_ids), invar["x"].shape[-1]*invar["t"].shape[-1]))

    return invar, outvar

def load_wf_fourier(h5f, model_ids, snapshot_ids, pred_key, data_type, test=False):
    model_ids.sort()
    invar = {}
    outvar = {}
    coords = h5f["element_coords_sz"][:,:]
    if test:
        invar['x'] = coords[None, :, 0].repeat(len(model_ids), 0) / 1e3
        invar['z'] = coords[None, :, 1].repeat(len(model_ids), 0)  / 1e3
        time = np.array([0.0744022, 0.2740638, 0.4737254, 0.673387, 0.8730486, 1.0727102, 1.2723718, 
                    1.4720334, 1.671695, 1.8713566, 2.0710182, 2.2706798, 2.4703414, 2.670003, 2.8696646])

    # invar['t'] = time
    invar['h'] = h5f['harmonics'][model_ids]
    outvar[pred_key] = h5f["X_coef"][model_ids][:,snapshot_ids] * 1e5
        
    if data_type == 'vector':
        for key in invar.keys():
            invar[key] = invar[key].reshape(len(model_ids), -1)
        for key in outvar.keys():
            outvar[key] = outvar[key].reshape(len(model_ids), -1)
    elif data_type == 'point':
        invar['x'] = coords[None, :, 0].repeat(len(model_ids), 0)  / 1e3 # b, 3648
        invar['z'] = coords[None, :, 1].repeat(len(model_ids), 0)  / 1e3 # b, 3648
        time = np.array([0.0744022, 0.2740638, 0.4737254, 0.673387, 0.8730486, 1.0727102, 1.2723718, 
                    1.4720334, 1.671695, 1.8713566, 2.0710182, 2.2706798, 2.4703414, 2.670003, 2.8696646])
        invar['t'] = time[snapshot_ids][None, :].repeat(len(model_ids), axis=0) # b, 4
        for key in invar.keys():
            if key == 'h':
                invar[key] = invar[key].reshape(len(model_ids), -1) # b, 405
        for key in outvar.keys():
            outvar[key] = outvar[key].reshape(len(model_ids), len(snapshot_ids)*3648, 16) # b, 4*3648, 16
    else:
        raise ValueError("data_type should be one of ['point', 'vector']")
    
    return invar, outvar

def fourier2phi(x, z, angle, fourier):
    # x (1, 1, 3000) z (1, 1, 3000) angle (1, 1, 3000) fourier (1, 1, 3000, 16)
    para_num = fourier.shape[-1]
    degree = para_num // 2 + 1

    fourier_ordered = np.zeros((fourier.shape[0], fourier.shape[1], x.shape[-1], degree), dtype='complex128')
    fourier_ordered[:, :, :, 0].real = fourier[:, :, :, 0]
    fourier_ordered[:, :, :, 1:degree].real = fourier[:, :, :, 1:para_num:2]
    fourier_ordered[:, :, :, 1:degree - 1].imag = fourier[:, :, :, 2:para_num:2]

    exp_array = np.exp(np.arange(1, degree) * 1j * angle[:, :, :, None])
    res = fourier_ordered[:, :, :, 0].copy()
    res += 2 * np.einsum('ijkl,ijkl->ijk', fourier_ordered[:, :, :, 1:degree], exp_array)
    
    return res.real

class HDF5GridDataset(Dataset):
    """lazy-loading HDF5 dataset"""
    auto_collation = True

    def __init__(
        self,
        filename: Union[str, Path],
        valid_idx,
        invar_keys: List[str],
        outvar_keys: List[str],
        lambda_weighting: Dict[str, float],
        data_type: str,
    ):
        self.valid_idx = valid_idx
        self._invar_keys = invar_keys
        self._outvar_keys = outvar_keys
        self.lambda_weighting = lambda_weighting
        self.path = Path(filename)
        self.data_type = data_type

        # check path
        assert self.path.is_file(), f"Could not find file {self.path}"
        assert self.path.suffix in [
            ".h5",
            ".hdf5",
        ], f"File type should be HDF5, got {self.path.suffix}"

        # check dataset/ get length
        # with h5py.File(self.path, "r") as f:
        #     self.f = f
        #     self.baseline = f['LatinSphericalHarmonicsAcousticBall0000']['disp'][:,:,2,:]
        self.baseline = None
        self.length = len(valid_idx)

    def __getitem__(self, idx):
        # invar, outvar = load_seis(self.f, f'{self.model_name}{self.valid_idx[idx]:0>4d}', self.outvar_keys[0], self.data_type, self.baseline)
        invar, outvar = load_seis(self.f, idx, self.outvar_keys[0], self.data_type, self.baseline)
        invar = Dataset._to_tensor_dict(
            {key: invar[key] for key in self._invar_keys}
        )
        outvar = Dataset._to_tensor_dict(
            {key: outvar[key] for key in self._outvar_keys}
        )
        lambda_weighting = Dataset._to_tensor_dict(
            {k: v*np.ones_like(outvar[k]) for k, v in self.lambda_weighting.items()}
        )
        return invar, outvar, lambda_weighting

    def __len__(self):
        return self.length

    def worker_init_fn(self, iworker):
        super().worker_init_fn(iworker)
        # open file on worker thread
        # note each torch DataLoader worker process should open file individually when reading
        # do not share open file descriptors across separate workers!
        # note files are closed when worker process is destroyed so no need to explicitly close
        self.f = h5py.File(self.path, "r")
        
    @property
    def invar_keys(self):
        return list(self._invar_keys)

    @property
    def outvar_keys(self):
        return list(self._outvar_keys)
