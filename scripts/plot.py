from modulus.sym.utils.io.plotter import ValidatorPlotter
import numpy as np
from matplotlib import pyplot as plt

class WavePlotter(ValidatorPlotter):
    "Define custom validator plotting class"
    def __init__(self, t) -> None:
        super().__init__()
        self.t = t
    def __call__(self, invar, true_outvar, pred_outvar):
        # filter invar to include only points y=0
        for key in invar.keys():
            if key in ["x", "y", "z"]:
                invar[key] = invar[key].reshape(1, 1, 16, -1)[0, 0].reshape(1,-1)
        for key in true_outvar.keys():
            true_outvar[key] = true_outvar[key].reshape(1, invar['t'].shape[1], 16, -1)[0, self.t].reshape(1,-1)
        for key in pred_outvar.keys():
            pred_outvar[key] = pred_outvar[key].reshape(1, invar['t'].shape[1], 16, -1)[0, self.t].reshape(1,-1)
        mask = np.abs(invar["y"]) ==0 # <=1e-6
        new_invar = {}
        new_true_outvar = {}
        new_pred_outvar = {}
        for key in invar.keys():
            if key in ["x", "z"]:
                new_invar[key] = invar[key][mask][:,None]
        for key in true_outvar.keys():
            new_true_outvar[key] = true_outvar[key][mask][:,None]
        for key in pred_outvar.keys():
            new_pred_outvar[key] = pred_outvar[key][mask][:,None]
        fs = super().__call__(new_invar, new_true_outvar, new_pred_outvar)
        return fs
    
class SeismoPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        # only consider one slice
        mask = np.abs(invar["zs"]) == 0
        new_invar = {}
        new_true_outvar = {}
        new_pred_outvar = {}
        for key in invar.keys():
            if key in ["ys", "t"]:
                new_invar[key] = invar[key][mask][:,None]
        for key in true_outvar.keys():
            new_true_outvar[key] = true_outvar[key][mask][:,None]
        for key in pred_outvar.keys():
            new_pred_outvar[key] = pred_outvar[key][mask][:,None]
        fs = super().__call__(new_invar, new_true_outvar, new_pred_outvar)
        return fs

class VelocityPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        fs = []
        for k in pred_outvar.keys():
            # Reshape the data for each key from 405 to 81x5
            reshaped_true = true_outvar[k].reshape(81, 5)
            reshaped_pred = pred_outvar[k].reshape(81, 5)
            diff = reshaped_true - reshaped_pred
            
            # Create a new figure for each key
            f = plt.figure(figsize=(15, 4), dpi=100)  # Adjust size as needed
            for i, (data, title) in enumerate(zip([reshaped_true, reshaped_pred, diff], ['True', 'Predicted', 'Difference'])):
                ax = plt.subplot(1, 3, i+1)
                heatmap = ax.imshow(data, cmap='viridis', aspect='auto')
                ax.set_title(f"{k} - {title}")
                plt.colorbar(heatmap, ax=ax)
            plt.tight_layout()
            fs.append((f, k))  # Append the figure and key as a tuple to the list
        return fs

def velocity_plotter(savepath, invar, tag='gt'):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import matplotlib.colors as colors
    from scipy.special import sph_harm

    def draw_map(m):
        # Draw a shaded-relief image
        m.shadedrelief(scale=0.2)
        # Lats and longs are returned as a dictionary
        lats = m.drawparallels(np.linspace(-90, 90, 5))
        lons = m.drawmeridians(np.linspace(-180, 180, 5))
        # Iterate through the returned dictionary items properly
        lat_lines = [line[0] for _, (line, _) in lats.items()]
        lon_lines = [line[0] for _, (line, _) in lons.items()]
        all_lines = lat_lines + lon_lines
        # Set the desired style
        for line in all_lines:
            line.set(linestyle="-", alpha=0.3, color='w')

    DepthList = [0., 200, 400, 600, 800]
    degree = 8
    # Fig Preparation
    for i, SlicingDepth in enumerate(DepthList):
        fig = plt.figure(figsize=(3.5, 3), dpi=200)
        ax = fig.add_subplot(111)
        grid_lat = np.linspace(-90, 90, 181)
        grid_lon = np.linspace(-180, 180, 361)
        LON, LAT = np.meshgrid(grid_lon, grid_lat)
        value = invar['h'].reshape(-1, 5)[:, i]
        coeff = {SlicingDepth: {}}
        para_index = 0
        for l in range(0, degree + 1):
            for m in range(-l, l + 1):
                name = '%s_%s' % (l, m)
                coeff[SlicingDepth][name] = value[para_index]
                para_index += 1
        TomoSum = np.zeros([len(grid_lat), len(grid_lon)])

        for l in range(0, degree + 1):
            for m in range(-l, l + 1):
                name = '%s_%s' % (l, m)
                Y_grid = sph_harm(m, l, np.radians(LON - 180), np.radians(90 - LAT))

                if m < 0:
                    Y_grid = np.sqrt(2) * (-1)**(-m) * Y_grid.imag
                elif m > 0:
                    Y_grid = np.sqrt(2) * (-1)**m * Y_grid.real

                TomoSum[:,:] = TomoSum[:,:] + coeff[SlicingDepth][name] * Y_grid

        m = Basemap(projection='moll', lon_0=0, resolution='l') # Mollweide Projection
        PLOT = m.pcolormesh(LON, LAT, TomoSum, latlon=True, cmap=plt.get_cmap('jet'), vmin=-0.1, vmax=0.1)
        # cbar = plt.colorbar(PLOT, ax=ax, shrink=0.5)
        # ax.set_title('Depth Slice at %s m to degrees %d' % (SlicingDepth, degree))
        draw_map(m)
        m.drawcoastlines(linewidth=0.1)
        plt.savefig(f'{savepath}/{tag}_{SlicingDepth}.png', dpi=600, bbox_inches='tight', transparent=True)
