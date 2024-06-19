import torch
import torch.backends.cudnn
import os
import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.hydra.config import ModulusConfig
import modulus
from modulus.sym.key import Key
import numpy as np
import h5py
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from scripts.load_data import load_seis, load_wf
from scripts.misc import (
    NormalizeX,
    NormalizeY,
    NormalizeZ,
    NormalizeT,
    NormalizeH,
)
from scripts.meta_info import except_models, num_models

from scripts.model import ModDeepONetArch
from scripts.plot import velocity_plotter, WavePlotter, SeismoPlotter

torch.random.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

tf_dt = torch.float32
np_dt = np.float32

torch.set_default_dtype(tf_dt)

wave_type = 'acoustic'
output_type = 'wfslice'
model_name = 'MLP'
model_path = f"./outputs/{wave_type}_{output_type}_{model_name}/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vis = False

def norm_data(invar, outvar, data_type='point'):
    for key in invar.keys():
        invar[key] = torch.from_numpy(invar[key]).to(device).to(tf_dt)
    if data_type == 'point':
        point_num = invar['x'].shape[-1]
        for key in invar.keys():
            if key in ['x', 'y', 'z', 'xs', 'ys', 'zs']:
                invar[key] = invar[key][:,None,:].repeat(1, invar['t'].shape[-1], 1).reshape(-1, 1)
            if key == 't':
                invar[key] = invar[key][:,:,None].repeat(1, 1, point_num).reshape(-1, 1)
    Nx, Ny, Nz, Nt, Nh = NormalizeX(), NormalizeY(), NormalizeZ(), NormalizeT(), NormalizeH()
    invar["xin"] = Nx(invar)["xin"]
    invar["yin"] = Ny(invar)["yin"]
    invar["zin"] = Nz(invar)["zin"]
    invar["tin"] = Nt(invar)["tin"]
    invar["hin"] = Nh(invar)["hin"]
    for key in outvar.keys():
            if len(outvar[key].shape) == 3:
                outvar[key] = outvar[key].reshape(-1, outvar[key].shape[-1])
    for key in invar.keys():
        if key in ['xin', 'yin', 'zin', 'tin', 'hin']:
            invar[key].requires_grad = True
    return invar, outvar

@modulus.sym.main(config_path=model_path, config_name="config")
def infer_wf(cfg: ModulusConfig) -> None:
    cfg.custom.data_dir = f'../data/{wave_type}'
    os.chdir("../../")
    #############################################
    #           1. Config network               #
    #############################################
    # define networks
    if cfg.custom.model == 'ModDeepONet':
        data_type = 'point'
        trunk_net = instantiate_arch(
            cfg=cfg.arch.trunk,
            input_keys=[Key("xin"), Key("yin"), Key("zin"), Key("tin")],
            weight_norm=True,
        )
        branch_net = instantiate_arch(
            cfg=cfg.arch.branch,
            weight_norm=True,
        )
        wave_net = ModDeepONetArch(
            branch_net=branch_net,
            trunk_net=trunk_net,
            output_keys=[Key(cfg.arch.deeponet.output_keys)],
            detach_keys=[],
        )
    elif cfg.custom.model == 'MLP':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.fully_connected,
        )
    elif cfg.custom.model == 'HighwayFourier':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.highway_fourier,
        )
    wave_net.load_state_dict(torch.load(os.path.join(model_path, "wave_network.0.pth")))
    wave_net.eval()
    wave_net.to(device)

    #############################################
    #               2. Load data                #
    #############################################
    val_models = list(range(0, int(0.1*num_models[wave_type])))
    val_models = list(set(val_models) - set(except_models[cfg.custom.data_dir.split('/')[-1]]))
    val_models = list(set(val_models) - set(except_models))
    times = [1,3,5,7,9,11,13]
    all_loss = []
    all_R = []
    all_time = []
    all_gt = []
    all_pred = []
    for m in tqdm(val_models):
        # load gt model and wf
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.wf_file), "r")
        invar, outvar = load_wf(h5f, np.array([m]), times, cfg.custom.pred_key, data_type, test=True)
        invar, outvar = norm_data(invar, outvar)
        if 'elastic' in cfg.custom.data_dir:
            source_h5f = h5py.File(os.path.join(cfg.custom.data_dir, 'elastic_source.h5'), "r")
            source = source_h5f['source'][m] / 1e10
            source = torch.tensor(source, dtype=tf_dt, device=device).reshape(1, -1)
            invar['hin'] = torch.concat([invar['hin'], source], dim=1)
        t1 = time.time()
        pred = wave_net(invar)
        t2 = time.time()
        all_time.append(t2-t1)
        pred = pred[cfg.custom.pred_key].detach().cpu().numpy()
        gt = outvar[cfg.custom.pred_key]
        all_gt.append(gt)
        all_pred.append(pred)
        # relative loss
        loss = np.sqrt(np.mean(np.square(gt - pred)) / np.var(gt))
        all_loss.append(loss)
        r = np.corrcoef(gt.reshape(-1), pred.reshape(-1))[0,1]
        all_R.append(r)
        if vis:
            save_dir = os.path.join(model_path, 'wf', f'{m}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for key in invar.keys():
                invar[key] = invar[key].detach().cpu().numpy()
            for t in range(len(times)):
                plotter = WavePlotter(t=t)
                fs = plotter(invar, {cfg.custom.pred_key: gt}, {cfg.custom.pred_key: pred})
                figure = fs[0][0]
                figure.savefig(f"{save_dir}/wf_{t}.png")
    all_gt = np.array(all_gt).reshape(len(val_models), len(times), -1)
    all_pred = np.array(all_pred).reshape(len(val_models), len(times), -1)

    print('mean std loss', np.mean(all_loss), np.std(all_loss))
    print('mean std r', np.mean(all_R), np.std(all_R))
    print('mean std time', np.mean(all_time), np.std(all_time))

@modulus.sym.main(config_path=model_path, config_name="config")
def infer_seis(cfg: ModulusConfig) -> None:
    cfg.custom.data_dir = f'../data/{wave_type}' # acoustic elastic
    os.chdir("../../")
    #############################################
    #           1. Config network               #
    #############################################
    # define networks
    if cfg.custom.model == 'ModDeepONet':
        data_type = 'point'
        trunk_net = instantiate_arch(
            cfg=cfg.arch.trunk,
            input_keys=[Key("xin"), Key("yin"), Key("zin"), Key("tin")],
            weight_norm=True,
        )
        branch_net = instantiate_arch(
            cfg=cfg.arch.branch,
            weight_norm=True,
        )
        wave_net = ModDeepONetArch(
            branch_net=branch_net,
            trunk_net=trunk_net,
            output_keys=[Key(cfg.arch.deeponet.output_keys)],
            detach_keys=[],
        )
    elif cfg.custom.model == 'MLP':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.fully_connected,
        )
    elif cfg.custom.model == 'HighwayFourier':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.highway_fourier,
        )
    wave_net.load_state_dict(torch.load(os.path.join(model_path, "wave_network.0.pth")))
    wave_net.eval()
    wave_net.to(device)

    #############################################
    #               2. Load data                #
    #############################################
    val_models = list(range(0, int(0.1*num_models[wave_type])))
    val_models = list(set(val_models) - set(except_models[cfg.custom.data_dir.split('/')[-1]]))
    all_loss = []
    all_R = []
    all_time = []
    for m in tqdm(val_models):
        # load gt model and wf
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.seis_file), "r")
        if 'diff' in cfg.custom.name:
            baseline = h5f['disp'][0:1][:,:,:,2,:]
        else:
            baseline = None
        invar, outvar = load_seis(h5f, np.array([m]), 'disp_z', data_type, norm=False, baseline=baseline, test=True)
        if m == 0:
            invar['h'] *= 0
        invar, outvar = norm_data(invar, outvar, data_type)
        if 'elastic' in cfg.custom.data_dir:
            source_h5f = h5py.File(os.path.join(cfg.custom.data_dir, 'elastic_source.h5'), "r")
            source = source_h5f['source'][m] / 1e10
            source = torch.tensor(source, dtype=tf_dt, device=device).reshape(1, -1)
            invar['hin'] = torch.concat([invar['hin'], source], dim=1)
        t1 = time.time()
        pred = wave_net(invar)
        t2 = time.time()
    
        gt = outvar['disp_z']
        pred = pred['disp_z'].detach().cpu().numpy()
        loss = np.sqrt(np.mean(np.square(gt - pred)) / np.var(gt))
        r = np.corrcoef(gt.reshape(-1), pred.reshape(-1))[0,1]
        mse = np.mean(np.square(gt - pred))
        all_loss.append(loss)
        all_R.append(r)
        all_time.append(t2-t1)
        if vis:
            save_dir = os.path.join(model_path, 'seis', f'{m}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for key in ['x', 'y', 'z', 't', 'xs', 'ys', 'zs']:
                invar[key] = invar[key].detach().cpu().numpy()
            plotter = SeismoPlotter()
            fs = plotter(invar, outvar, {'disp_z': pred})
            figure = fs[0][0]
            figure.savefig(os.path.join(save_dir,f'seis.png'))
    print('mean std loss', np.mean(all_loss), np.std(all_loss))
    print('mean std r', np.mean(all_R), np.std(all_R))
    print('mean std time', np.mean(all_time), np.std(all_time))

@modulus.sym.main(config_path=model_path, config_name="config")
def infer_velocity(cfg: ModulusConfig) -> None:
    os.chdir("../../")
    #############################################
    #           1. Config network               #
    #############################################
    # define networks
    if cfg.custom.model == 'ModDeepONet':
        data_type = 'point'
        trunk_net = instantiate_arch(
            cfg=cfg.arch.trunk,
            input_keys=[Key("xin"), Key("zin"), Key("tin")],
            weight_norm=True,
        )
        branch_net = instantiate_arch(
            cfg=cfg.arch.branch,
            weight_norm=True,
        )
        wave_net = ModDeepONetArch(
            branch_net=branch_net,
            trunk_net=trunk_net,
            output_keys=[Key(cfg.arch.deeponet.output_keys)],
            detach_keys=[],
        )
    elif cfg.custom.model == 'MLP':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.fully_connected,
        )
    wave_net.load_state_dict(torch.load(os.path.join(model_path, "wave_network.0.pth")))
    wave_net.eval()
    wave_net.to(device)

    #############################################
    #               2. Load data                #
    #############################################
    val_models = list(range(0, int(0.1*num_models[wave_type])))
    val_models = list(set(val_models) - set(except_models[cfg.custom.data_dir.split('/')[-1]]))
    all_r = []
    all_rl2 = []
    for m in tqdm(val_models):
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.seis_file), "r")
        outvar_seis_val, invar_seis_val = load_seis(h5f, np.array([m]), 'disp_z', data_type, norm=True, baseline=None, test=True, transpose=False)
        torch_invar_seis = {}
        for key in invar_seis_val.keys():
            torch_invar_seis[key] = torch.from_numpy(invar_seis_val[key]).to(device).to(tf_dt)
        pred = wave_net(torch_invar_seis)
        pred = pred['h'].detach().cpu().numpy()
        gt = outvar_seis_val['h']
        # relative loss
        loss = np.sqrt(np.mean(np.square(gt - pred)) / np.var(gt))
        all_rl2.append(loss)
        r = np.corrcoef(gt.reshape(-1), pred.reshape(-1))[0,1]
        all_r.append(r)
        if vis:
            save_path = f"{model_path}/velocity/{m}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            velocity_plotter(savepath=save_path, invar={'h': pred}, tag='pred')
            velocity_plotter(savepath=save_path, invar={'h': gt}, tag='gt')

            # draw parameters as points to show correlation
            plt.figure(figsize=(6,6))
            fontsize = 18
            plt.scatter(gt.reshape(-1), pred.reshape(-1), color='#36A9D3', alpha=0.9)
            # do linear regression and draw a line to fit the points with minial square error]
            x = gt.reshape(-1)
            y = pred.reshape(-1)
            A = np.vstack([x, np.ones(len(x))]).T
            a, c = np.linalg.lstsq(A, y, rcond=None)[0]
            se = np.array([-0.04, 0.04])
            plt.plot(se, a*se + c, color='gray')
            plt.xlabel('Ground truth structure harmonics', fontsize=fontsize)
            plt.ylabel('Inverted structure harmonics', fontsize=fontsize)
            plt.xticks([-0.04, -0.02, 0, 0.02, 0.04], fontsize=fontsize)
            plt.yticks([-0.04, -0.02, 0, 0.02, 0.04], fontsize=fontsize)
            plt.xlim(-0.04, 0.04)
            plt.ylim(-0.04, 0.04)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            r = np.corrcoef(x, y)[0,1]
            plt.text(0.05, 0.9, f"r = {r:.3f}", fontsize=fontsize, transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))
            plt.savefig(f"{model_path}/velocity/{m}_/correlation.png", dpi=600, bbox_inches='tight')
            # calculate correlation for each depth
            for i in range(5):
                plt.figure(figsize=(6,6))
                fontsize = 24
                x = gt.reshape(81, 5)[:,i]
                y = pred.reshape(81, 5)[:,i]
                A = np.vstack([x, np.ones(len(x))]).T
                a, c = np.linalg.lstsq(A, y, rcond=None)[0]
                se = np.array([-0.04, 0.04])
                plt.plot(se, a*se + c, color='gray')
                plt.scatter(x, y, color='#36A9D3', alpha=0.9)
                plt.xlabel('Ground truth', fontsize=fontsize)
                plt.ylabel('Inverted', fontsize=fontsize)
                plt.xticks([-0.04, -0.02, 0, 0.02, 0.04], fontsize=fontsize)
                plt.yticks([-0.04, -0.02, 0, 0.02, 0.04], fontsize=fontsize)
                plt.xlim(-0.04, 0.04)
                plt.ylim(-0.04, 0.04)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                r = np.corrcoef(x, y)[0,1]
                plt.text(0.05, 0.9, f"r = {r:.3f}", fontsize=fontsize, transform=plt.gca().transAxes, 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))
                plt.savefig(f"{model_path}/velocity/{m}/R_{i}.png", dpi=600, bbox_inches='tight')
    all_rl2 = np.array(all_rl2)
    print('mean rl2', all_rl2.mean(), 'std rl2', all_rl2.std())
    all_r = np.array(all_r)
    print('mean r', all_r.mean(), 'std r', all_r.std())

def infer_wf_mean_model(wave_type):
    data_dir = f'../data/{wave_type}'
    wf_file = 'wf_slice_data.h5'
    h5f = h5py.File(os.path.join(data_dir, wf_file), "r")
    snapshots = [1,3,5,7,9,11,13]
    save_path = f'outputs/{wave_type}_mean_wf.npy'
    if not os.path.exists(save_path):
        train_models = list(range(int(0.1*num_models[wave_type]), num_models[wave_type]))
        train_models = list(set(train_models) - set(except_models[data_dir.split('/')[-1]]))
        # load all wf
        all_wf = h5f['X'][train_models][:,snapshots]
        mean_wf = all_wf.mean(0)
        # save the mean wf into a npy file
        np.save(save_path, mean_wf)
    else:
        mean_wf = np.load(save_path)
    # evaluate the mean model
    val_models = list(range(0, int(0.1*num_models[wave_type])))
    val_models = list(set(val_models) - set(except_models[data_dir.split('/')[-1]]))
    times = [1,3,5,7,9,11,13]
    all_loss = []
    all_R = []
    all_gt = []
    all_pred = []
    for m in tqdm(val_models):
        wf = h5f['X'][m][snapshots]
        loss = np.sqrt(np.mean(np.square(wf - mean_wf)) / np.var(wf))
        all_loss.append(loss)
        r = np.corrcoef(wf.reshape(-1), mean_wf.reshape(-1))[0,1]
        all_R.append(r)
        all_gt.append(wf)
        all_pred.append(mean_wf)

    all_gt = np.array(all_gt).reshape(len(val_models), len(times), -1)
    all_pred = np.array(all_pred).reshape(len(val_models), len(times), -1)
    print('mean std loss', np.mean(all_loss), np.std(all_loss))
    print('mean std r', np.mean(all_R), np.std(all_R))

def infer_seis_mean_model(wave_type):
    data_dir = f'../data/{wave_type}'
    seis_file = 'seis_data.h5'
    h5f = h5py.File(os.path.join(data_dir, seis_file), "r")
    save_path = f'outputs/{wave_type}_mean_seis.npy'
    if not os.path.exists(save_path):
        train_models = list(range(int(0.1*num_models[wave_type]), num_models[wave_type]))
        train_models = list(set(train_models) - set(except_models[data_dir.split('/')[-1]]))
        # load all seis
        all_seis = h5f['disp'][:,:,:,2,:][train_models]
        mean_seis = all_seis.mean(0)
        # save the mean seis into a npy file
        np.save(save_path, mean_seis)
    else:
        mean_seis = np.load(save_path)
    # evaluate the mean model
    val_models = list(range(0, int(0.1*num_models[wave_type])))
    val_models = list(set(val_models) - set(except_models[data_dir.split('/')[-1]]))
    all_loss = []
    all_R = []
    for m in tqdm(val_models):
        seis = h5f['disp'][m][:,:,2,:]
        loss = np.sqrt(np.mean(np.square(seis - mean_seis)) / np.var(seis))
        all_loss.append(loss)
        r = np.corrcoef(seis.reshape(-1), mean_seis.reshape(-1))[0,1]
        all_R.append(r)
    print('mean std loss', np.mean(all_loss), np.std(all_loss))
    print('mean std r', np.mean(all_R), np.std(all_R))

if __name__ == '__main__':
    if output_type == 'wfslice':
        infer_wf()
    elif output_type == 'seis':
        infer_seis()
    elif output_type == 'velocity':
        infer_velocity()
    else:
        raise ValueError(f"output_type {output_type} is not supported")
    # infer_wf_mean_model(wave_type=wave_type)
    # infer_seis_mean_model(wave_type=wave_type)
    