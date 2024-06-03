from torch import Tensor
from typing import Dict
import torch.nn as nn
import numpy as np
import torch
import torch.backends.cudnn
from modulus.sym.loss import Loss, PointwiseLossNorm
from modulus.sym.domain.constraint import Constraint
from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.constants import TF_SUMMARY
from typing import Dict, List
from modulus.sym.loss import Loss, PointwiseLossNorm

torch.random.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Parallel_DeepONetConstraint(DeepONetConstraint):
    def load_data(self):
        # get train points from dataloader
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        # x: (b, point_num) z: (b, point_num) h: (b, 405) t: (b, t) phi: (b, point_num*t, 16)
        point_num = invar['x'].shape[-1]
        for key in invar.keys():
            if key in ['x', 'y', 'z', 'vp', 'xs', 'ys', 'zs']:
                invar[key] = invar[key][:,None,:].repeat(1, invar['t'].shape[-1], 1).reshape(-1, 1)
        invar['t'] = invar['t'][:,:,None].repeat(1, 1, point_num).reshape(-1, 1)
        for key in true_outvar.keys():
            if len(true_outvar[key].shape) == 3:
                true_outvar[key] = true_outvar[key].reshape(-1, true_outvar[key].shape[-1])
        for key in lambda_weighting.keys():
            if len(lambda_weighting[key].shape) == 3:
                lambda_weighting[key] = lambda_weighting[key].reshape(-1, lambda_weighting[key].shape[-1])
        self._input_vars = Constraint._set_device(
            invar, device=self.device, requires_grad=True
        )
        self._target_vars = Constraint._set_device(true_outvar, device=self.device)
        self._lambda_weighting = Constraint._set_device(
            lambda_weighting, device=self.device
        )

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            point_num = invar['x'].shape[-1]
            for key in invar.keys():
                if key in ['x', 'y', 'z', 'vp', 'xs', 'ys', 'zs']:
                    invar[key] = invar[key][:,None,:].repeat(1, invar['t'].shape[-1], 1).reshape(-1, 1)
            invar['t'] = invar['t'][:,:,None].repeat(1, 1, point_num).reshape(-1, 1)
            for key in true_outvar.keys():
                if len(true_outvar[key].shape) == 3:
                    true_outvar[key] = true_outvar[key].reshape(-1, true_outvar[key].shape[-1])
            for key in lambda_weighting.keys():
                if len(lambda_weighting[key].shape) == 3:
                    lambda_weighting[key] = lambda_weighting[key].reshape(-1, lambda_weighting[key].shape[-1])
            # Set grads to false here for inputs, static var has allocation already
            input_vars = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(true_outvar, device=self.device)
            lambda_weighting = Constraint._set_device(
                lambda_weighting, device=self.device
            )
            for key in input_vars.keys():
                self._input_vars[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])
            for key in lambda_weighting.keys():
                self._lambda_weighting[key].copy_(lambda_weighting[key])

class Parallel_PointwiseValidator(PointwiseValidator):
    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            point_num = invar0['x'].shape[-1]
            for key in invar0.keys():
                if key in ['x', 'y', 'z', 'vp', 'xs', 'ys', 'zs']:
                    invar0[key] = invar0[key][:,None,:].repeat(1, invar0['t'].shape[-1], 1).reshape(-1, 1)
            invar0['t'] = invar0['t'][:,:,None].repeat(1, 1, point_num).reshape(-1, 1)
            for key in true_outvar0.keys():
                if len(true_outvar0[key].shape) == 3:
                    true_outvar0[key] = true_outvar0[key].reshape(-1, true_outvar0[key].shape[-1])
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        # TODO add metrics specific for validation
        # TODO: add potential support for lambda_weighting
        losses = PointwiseValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)
        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:
            var_to_polyvtk(
                {**invar, **named_true_outvar, **named_pred_outvar}, results_dir + name
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True
                )
        return losses

class LazyConstraint(DeepONetConstraint):
    @classmethod
    def from_numpy(
        cls,
        nodes: List[None],
        batch_size: int,
        dataset: None,
        loss: Loss = PointwiseLossNorm,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):

        return cls(
            nodes=nodes,
            batch_size=batch_size,
            dataset=dataset,
            loss=loss,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    
class Parallel_LazyConstraint(DeepONetConstraint):
    def load_data(self):
        # get train points from dataloader
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        for key in invar.keys():
            if len(invar[key].shape) == 3:
                invar[key] = invar[key].reshape(-1, invar[key].shape[-1])
        for key in true_outvar.keys():
            if len(true_outvar[key].shape) == 3:
                true_outvar[key] = true_outvar[key].reshape(-1, true_outvar[key].shape[-1])
        for key in lambda_weighting.keys():
            if len(lambda_weighting[key].shape) == 3:
                lambda_weighting[key] = lambda_weighting[key].reshape(-1, lambda_weighting[key].shape[-1])
        self._input_vars = Constraint._set_device(
            invar, device=self.device, requires_grad=True
        )
        self._target_vars = Constraint._set_device(true_outvar, device=self.device)
        self._lambda_weighting = Constraint._set_device(
            lambda_weighting, device=self.device
        )

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            for key in invar.keys():
                invar[key] = invar[key].reshape(-1, invar[key].shape[-1])
            for key in true_outvar.keys():
                true_outvar[key] = true_outvar[key].reshape(-1, true_outvar[key].shape[-1])
            for key in lambda_weighting.keys():
                lambda_weighting[key] = lambda_weighting[key].reshape(-1, lambda_weighting[key].shape[-1])
            # Set grads to false here for inputs, static var has allocation already
            input_vars = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(true_outvar, device=self.device)
            lambda_weighting = Constraint._set_device(
                lambda_weighting, device=self.device
            )
            for key in input_vars.keys():
                self._input_vars[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])
            for key in lambda_weighting.keys():
                self._lambda_weighting[key].copy_(lambda_weighting[key])
    @classmethod
    def from_numpy(
        cls,
        nodes: List[None],
        batch_size: int,
        dataset: None,
        loss: Loss = PointwiseLossNorm,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):

        return cls(
            nodes=nodes,
            batch_size=batch_size,
            dataset=dataset,
            loss=loss,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

class NormalizeX(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"xin": in_vars["x"]}

class NormalizeY(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"yin": in_vars["y"]}

class NormalizeZ(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"zin": in_vars["z"]}

class NormalizeT(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"tin": in_vars["t"]  / 2}

class NormalizeH(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"hin": in_vars["h"] * 4e1}

    
def cartesian_to_spherical(x, y, z, R):
    r = torch.sqrt(x**2 + y**2 + z**2)
    # -π/2 to π/2
    latitude = torch.asin(z / r)
    longitude = torch.atan2(y, x)
    
    depth = R - r
    
    latitude = latitude * (180 / torch.pi)
    longitude = longitude * (180 / torch.pi)

    return latitude, longitude, depth

def spherical_to_cartesian(latitude, longitude, depth, R):
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)

    r = R - depth

    x = r * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = r * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = r * np.sin(latitude_rad)
    
    return x, y, z
