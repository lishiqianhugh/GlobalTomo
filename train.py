import torch
import torch.backends.cudnn
import numpy as np
import h5py
import os
import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.node import Node
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.loss import PointwiseLossNorm
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint.continuous import (
    DeepONetConstraint,
    PointwiseConstraint,
)

from scripts.model import ModDeepONetArch
from scripts.eq import FreeSurface, WaveEquation
from scripts.plot import WavePlotter, SeismoPlotter, VelocityPlotter
from scripts.load_data import HDF5GridDataset, load_seis, load_wf, load_interior, load_surface
from scripts.misc import Parallel_DeepONetConstraint, Parallel_PointwiseValidator, LazyConstraint, Parallel_LazyConstraint
from scripts.misc import (
    NormalizeX,
    NormalizeY,
    NormalizeZ,
    NormalizeT,
    NormalizeH,
)
from scripts.meta_info import except_models

torch.random.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

tf_dt = torch.float32
np_dt = np.float32

torch.set_default_dtype(tf_dt)

use_wandb = False
local_rank = 0 # int(os.environ.get("SLURM_LOCALID"))
if use_wandb and local_rank == 0:
    import wandb
    wandb.login(key='', relogin=True)
    wandb.init(
        project="seismic", sync_tensorboard=True, save_code=True, mode="online"
    )


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    if use_wandb and local_rank == 0:
        wandb.run.name = cfg.custom.name
    os.chdir("../../")
    # set the network directory for checkpoints
    cfg.network_dir = f"outputs/{cfg.custom.name}"
    if not os.path.exists(cfg.network_dir):
        os.makedirs(cfg.network_dir)
    assert os.path.exists(f'conf/{cfg.custom.name}.yaml')
    # copy the original config file into network dir
    os.system(f"cp conf/{cfg.custom.name}.yaml {cfg.network_dir}/config.yaml")
    #############################################
    #    1. Config domain, PDEs, and network    #
    #############################################
    # define the prediction variable
    pred_key = cfg.custom.pred_key
    # define PDEs for physics-informed training
    we = WaveEquation("phi")
    bc = FreeSurface("phi")
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
    elif cfg.custom.model == 'Fourier':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.fourier,
        )
    elif cfg.custom.model == 'HighwayFourier':
        data_type = 'vector'
        wave_net = instantiate_arch(
            cfg=cfg.arch.highway_fourier,
        )

    domain = Domain()

    #############################################
    #               2. Load data                #
    #############################################
    train_models = list(range(cfg.custom.train_models[0], cfg.custom.train_models[1]))
    val_models = list(range(cfg.custom.val_models[0], cfg.custom.val_models[1]))
    train_models = list(set(train_models) - set(except_models[cfg.custom.data_dir.split('/')[-1]]))
    val_models = list(set(val_models) - set(except_models[cfg.custom.data_dir.split('/')[-1]]))
    train_shots = cfg.custom.train_shots
    val_shots = cfg.custom.val_shots

    nodes = (
        [
            Node(["h"], ["hin"], NormalizeH()),
            Node(["x"], ["xin"], NormalizeX()),
            Node(["y"], ["yin"], NormalizeY()),
            Node(["z"], ["zin"], NormalizeZ()),
            Node(["t"], ["tin"], NormalizeT()),
        ] +
        [
            wave_net.make_node(name="wave_network"),
        ] 
        # +
        # we.make_nodes(detach_names=["vs", "vp"]) +
        # bc.make_nodes()
    )

    #############################################
    #           3. Add constraints              #
    #############################################
    if cfg.custom.wavefield:
        print("###### Use Wavefield ######")
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.wf_file), "r")
        if cfg.custom.data_dir.split('/')[-1] == "elastic":
            source = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.source_file), "r")
        else:
            source = None
        # add wavefield validator
        for m in val_models:
            invar_wf_val, outvar_wf_val = load_wf(h5f, np.array([m]), val_shots, pred_key, data_type, source, test=True)
            for i in range(len(val_shots)):
                wf_val = "Parallel_PointwiseValidator" if data_type == "point" else "PointwiseValidator"
                wf_l2 = eval(wf_val)(
                    nodes=nodes,
                    invar=invar_wf_val,
                    true_outvar={pred_key: outvar_wf_val[pred_key]},
                    plotter=WavePlotter(t=i),
                    requires_grad=False,
                )
                domain.add_validator(wf_l2, f"wf_l2_{m:04d}_{val_shots[i]:04d}")
        # add wavefield constraint for training
        invar_wf, outvar_wf = load_wf(h5f, train_models, train_shots, pred_key, data_type, source)
        wf_constraint = "Parallel_DeepONetConstraint" if data_type == "point" else "DeepONetConstraint"
        wavefield = eval(wf_constraint).from_numpy(
            nodes=nodes,
            invar=invar_wf,
            outvar=outvar_wf,
            batch_size=cfg.batch_size.wavefield,
            num_workers=8,
            loss=PointwiseLossNorm()
        )
        domain.add_constraint(wavefield, f"wavefield")

    if cfg.custom.seismogram:
        print("###### Use seismogram ######")
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.seis_file), "r")
        if cfg.custom.data_dir.split('/')[-1] == "elastic":
            source = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.source_file), "r")
        else:
            source = None
        if cfg.custom.lazy_loading:
            for m in val_models:
                invar_seis_val, outvar_seis_val = load_seis(h5f, np.array([m]), pred_key, data_type, norm=False, 
                                                            baseline=None, source=source, test=True)
                seis_val = "Parallel_PointwiseValidator" if data_type == "point" else "PointwiseValidator" 
                seis_l2 = eval(seis_val)(
                    nodes=nodes,
                    invar=invar_seis_val,
                    true_outvar=outvar_seis_val,
                    plotter=SeismoPlotter(),
                    requires_grad=True,
                )
                domain.add_validator(seis_l2, f"seis_l2_{m:04d}")
            dataset = HDF5GridDataset(
                filename=cfg.custom.data_dir,
                valid_idx=train_models,
                invar_keys=["h"],
                outvar_keys=[pred_key],
                lambda_weighting={pred_key: cfg.custom.weight_seis},
                data_type=data_type,
            )
            seis_constraint = "Parallel_LazyConstraint" if data_type == "point" else "LazyConstraint"
            seis = eval(seis_constraint).from_numpy(
                nodes=nodes,
                dataset=dataset,
                batch_size=cfg.batch_size.seismogram,
                loss=PointwiseLossNorm(),
                num_workers=8,
            )
            domain.add_constraint(seis, f"seis")
        else:
            for m in val_models:
                invar_seis_val, outvar_seis_val = load_seis(h5f, np.array([m]), pred_key, data_type, norm=False, 
                                                            baseline=None, source=source, test=True)
                seis_val = "Parallel_PointwiseValidator" if data_type == "point" else "PointwiseValidator" 
                seis = eval(seis_val)(
                    nodes=nodes,
                    invar=invar_seis_val,
                    true_outvar={pred_key: outvar_seis_val[pred_key]},
                    plotter=SeismoPlotter(),
                    requires_grad=True,
                )
                domain.add_validator(seis, f"seis_{m:04d}")
            invar_seis, outvar_seis = load_seis(h5f, train_models, pred_key, data_type, norm=False, 
                                                baseline=None, source=source, test=False)
            # add seismogram constraint
            seis_constraint = "Parallel_DeepONetConstraint" if data_type == "point" else "DeepONetConstraint"
            seis = eval(seis_constraint).from_numpy(
                nodes=nodes,
                invar=invar_seis,
                outvar={pred_key: outvar_seis[pred_key]},
                batch_size=cfg.batch_size.seismogram,
                num_workers=8,
                loss=PointwiseLossNorm(),
                lambda_weighting={pred_key: cfg.custom.weight_seis*np.ones_like(outvar_seis[pred_key])},
            )
            domain.add_constraint(seis, f"seis")

    if cfg.custom.velocity:
        print("###### Use seismogram ######")
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.seis_file), "r")
        if cfg.custom.data_dir.split('/')[-1] == "elastic":
            source = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.source_file), "r")
        else:
            source = None
        for m in val_models:
            invar_seis_new, outvar_seis_new = load_seis(h5f, np.array([m]), pred_key, data_type, norm=True, baseline=None, transpose=False, test=True)
            seis_val = "Parallel_PointwiseValidator" if data_type == "point" else "PointwiseValidator" 
            seis = eval(seis_val)(
                nodes=nodes,
                invar={pred_key: outvar_seis_new[pred_key]},
                true_outvar={'h': invar_seis_new['h']},
                plotter=VelocityPlotter(),
                requires_grad=True,
            )
            domain.add_validator(seis, f"seis_{m:04d}")
        invar_seis, outvar_seis = load_seis(h5f, train_models, pred_key, data_type, norm=True, baseline=None, transpose=False, test=False)
        # add seismogram constraint
        seis_constraint = "Parallel_DeepONetConstraint" if data_type == "point" else "DeepONetConstraint"
        seis = eval(seis_constraint).from_numpy(
            nodes=nodes,
            invar={pred_key: outvar_seis[pred_key]},
            outvar={'h': invar_seis['h']},
            batch_size=cfg.batch_size.velocity,
            num_workers=4,
            loss=PointwiseLossNorm(),
            lambda_weighting={'h': np.ones_like(invar_seis['h'])},
        )
        domain.add_constraint(seis, f"seis")

    if cfg.custom.interior:
        print("###### Use Interior Constraint ######")
        # add interior constraint
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.interior_file), "r")
        invar_interior, outvar_interior = load_interior(h5f, train_models, val_shots)
        interior_constraint = "Parallel_DeepONetConstraint" if data_type == "point" else "DeepONetConstraint"
        interior = eval(interior_constraint).from_numpy(
            nodes=nodes,
            invar=invar_interior,
            outvar=outvar_interior,
            batch_size=cfg.batch_size.interior,
            num_workers=4,
            loss=PointwiseLossNorm(),
            lambda_weighting={"wave_equation": cfg.custom.weight_waveequation*np.ones_like(outvar_interior["wave_equation"])},
        )
        domain.add_constraint(interior, "Interior")

    if cfg.custom.surface:
        print("###### Use Free Surface Constraint ######")
        h5f = h5py.File(os.path.join(cfg.custom.data_dir, cfg.custom.surface_file), "r")
        invar_surf, outvar_surf = load_surface(h5f, train_models, val_shots)
        surf_constraint = "Parallel_DeepONetConstraint" if data_type == "point" else "DeepONetConstraint"
        surface = eval(surf_constraint).from_numpy(
            nodes=nodes,
            invar=invar_surf,
            outvar=outvar_surf,
            batch_size=cfg.batch_size.surface,
            num_workers=4,
            loss=PointwiseLossNorm(),
            lambda_weighting={"free_surface": cfg.custom.weight_freesurface*np.ones_like(outvar_surf["free_surface"])},
        )
        domain.add_constraint(surface, "Surface")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
    if use_wandb and local_rank == 0:
        wandb.finish()
