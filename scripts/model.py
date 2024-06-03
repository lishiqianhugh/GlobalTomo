from modulus.sym.models.arch import Arch
from modulus.sym.models.deeponet import DeepONetArch
from typing import Dict, List, Union
from modulus.sym.key import Key
import torch
from torch import Tensor

class ModDeepONetArch(DeepONetArch):
    def __init__(
        self,
        branch_net: Arch,
        trunk_net: Arch,
        output_keys: List[Key] = None,
        detach_keys: List[Key] = [],
        branch_dim: Union[None, int] = None,
        trunk_dim: Union[None, int] = None,
    ):
        super().__init__(branch_net, trunk_net, output_keys, detach_keys, branch_dim, trunk_dim)
        # self.output_linear = nn.Linear(self.deepo_dim, 16, bias=False)
    @staticmethod
    def concat_input(
        input_variables: Dict[str, Tensor],
        mask: List[str],
        detach_dict: Union[Dict[str, int], None] = None,
        dim: int = -1,
    ) -> Tensor:
        output_tensor = []
        for key in mask:
            if key == "hin":
                continue
            if detach_dict is not None and key in detach_dict:
                x = input_variables[key].detach()
            else:
                x = input_variables[key]
            output_tensor += [x]
        return torch.cat(output_tensor, dim=dim), input_variables["hin"]
    
    def _tensor_forward(self, x: Tensor, h: Tensor) -> Tensor:
        assert self.supports_func_arch, (
            f"The combination of branch_net {type(self.branch_net)} and trunk_net "
            + f"{type(self.trunk_net)} does not support FuncArch."
        )
        # branch_x = self.slice_input(x, self.branch_slice_index, dim=-1)
        # trunk_x = self.slice_input(x, self.trunk_slice_index, dim=-1)

        branch_x = h
        trunk_x = x
        branch_output = self.branch_net._tensor_forward(branch_x)
        trunk_output = self.trunk_net._tensor_forward(trunk_x)

        # Convert ouputs into 1D feature vectors
        if torch._C._functorch.is_gradtrackingtensor(
            trunk_output
        ) or torch._C._functorch.is_batchedtensor(trunk_output):
            # batched tensor does not have the original shape
            branch_output = branch_output.view(-1)
            trunk_output = trunk_output.view(-1)
        else:
            branch_output = branch_output.view(branch_output.shape[0], -1)
            trunk_output = trunk_output.view(trunk_output.shape[0], -1)

        assert (
            branch_output.size(-1) == self.branch_dim
        ), f"Invalid feature dimension from branch net, expected {self.branch_dim} but found {branch_output.size(-1)}"
        assert (
            trunk_output.size(-1) == self.trunk_dim
        ), f"Invalid feature dimension from trunk net, expected {self.trunk_dim} but found {trunk_output.size(-1)}"

        # Send through final linear layers
        branch_output = self.branch_linear(branch_output)
        trunk_output = self.trunk_linear(trunk_output)

        branch_output = branch_output.unsqueeze(1) # [3, 1, 1024]
        trunk_output = trunk_output.view(branch_output.shape[0], -1, trunk_output.shape[1]) # [3,10000, 1024]

        y = self.output_linear(branch_output * trunk_output)

        y = y.view(-1, y.shape[-1])

        y = self.process_output(y, self.output_scales_tensor)
        return y
    
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x, h = self.concat_input(
                in_vars,
                self.input_key_dict.keys(),
                detach_dict=self.detach_key_dict,
                dim=-1,
            )
        y = self._tensor_forward(x, h)
        return {list(self.output_key_dict.keys())[0]: y}
        # return self.split_output(y, self.output_key_dict, dim=-1)