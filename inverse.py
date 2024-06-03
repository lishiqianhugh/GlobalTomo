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

from scripts.load_data import load_seis
from scripts.misc import (
    NormalizeX,
    NormalizeY,
    NormalizeZ,
    NormalizeT,
    NormalizeH,
)
from scripts.meta_info import except_models

from scripts.model import ModDeepONetArch
from scripts.plot import velocity_plotter

torch.random.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

tf_dt = torch.float32
np_dt = np.float32

torch.set_default_dtype(tf_dt)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_path = "./outputs/checkpoints/acoustic/seis_MLP/"
val_models = [185]
start_num=10
vis = False
@modulus.sym.main(config_path=model_path, config_name="config")
def inverse(cfg: ModulusConfig) -> None:
    # name = 'rawseis_ModDeepONet_6_600' # 'rawseis_MLP_6_500' # 'rawseis_ModDeepONet_6_600_8gpu'
    cfg.custom.data_dir = '../data/acoustic/'
    cfg.network_dir = f"outputs/{cfg.custom.name}"
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
    wave_net.load_state_dict(torch.load(os.path.join(model_path, "wave_network.0.pth")))
    wave_net.eval()
    wave_net.to(device)

    #############################################
    #               2. Load data                #
    #############################################
    for m in tqdm(val_models):
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.seis_file), "r")
        invar, outvar = load_seis(h5f, np.array([m]), 'disp_z', data_type, norm=False, baseline=None, test=True)
        invar, outvar = norm_data(invar, outvar, data_type)
        gt_h = invar['hin']

        sample_num = 10000
        sampler = LatinHypercube(d=405, seed=0)
        latin_samples = sampler.random(n=sample_num)
        latin_samples = 2 * latin_samples - 1
        best_episode = 0
        best_r = 0
        for episode in tqdm(range(0, start_num)):
            start_model = latin_samples[episode]
            start_model = start_model.reshape(1, -1)
            invar['hin'] = torch.tensor(start_model).to(device).to(tf_dt)
            invar['hin'].requires_grad = True  

            optim = torch.optim.LBFGS(params=[invar['hin']], lr=8e-2, max_iter=5, history_size=30)
            loss_fn = torch.nn.L1Loss()

            def closure():
                optim.zero_grad()
                predicted_output = wave_net(invar)['disp_z']
                loss_seis = loss_fn(predicted_output, torch.from_numpy(outvar['disp_z']).to(device).to(tf_dt))
                loss_reg = 1e-3 * torch.norm(invar['hin'], 2)
                loss = loss_seis + loss_reg
                loss.backward()
                return loss

            for step in range(40):
                optim.step(closure)
                loss_seis = loss_fn(wave_net(invar)['disp_z'], torch.from_numpy(outvar['disp_z']).to(device).to(tf_dt))
                r = np.corrcoef(invar['hin'].detach().cpu().numpy().reshape(-1), gt_h.detach().cpu().numpy().reshape(-1))[0,1]
                if step % 1 == 0:
                    print('model id', m, 'episode', episode, 'step', step, 'loss', np.round(loss_seis.item(), 3), 'r', np.round(r, 3))

            if r > best_r:
                best_episode = episode
                best_r = r
            if vis:
                inverse_path = os.path.join(model_path, 'inv', f'{m}')
                if not os.path.exists(inverse_path):
                    os.makedirs(inverse_path)
                np_pred_h = invar['hin'].detach().cpu().numpy()/4e1
                np_gt_h = gt_h.detach().cpu().numpy()/4e1
                velocity_plotter(savepath=inverse_path, invar={'h': np_pred_h}, tag='pred')
                velocity_plotter(savepath=inverse_path, invar={'h': np_gt_h}, tag='gt') 
        print('best_episode', best_episode, 'best_r', best_r)

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

if __name__ == "__main__":
    inverse()
