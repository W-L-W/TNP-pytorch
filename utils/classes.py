from dataclasses import dataclass, field
from typing import Any, Dict
from jaxtyping import Float
from torch import Tensor
from utils.misc import AttrDict

StateDict = Dict[str, Any]
# this StateDict hint is explicity defined in optimizer file, but
# only implicitly in scheduler or nn.Module files

# short note on dimensions:
# B: batch size
# N: total number of points
# Nc: number of context points
# Nt: number of target points
# Dx: dimension of input
# Dy: dimension of output
# will use these throughout
# choice: for now use dim convention from codebase


@dataclass(frozen=True)
class Batch(AttrDict):
    """
    Batch of (x,y) pairs for context and target points.
    Note have identity $N = Nc + Nt$
    """

    x: Float[Tensor, "*B N Dx"]
    y: Float[Tensor, "*B N Dy"]
    Nc: int

    xc: Float[Tensor, "*B Nc Dx"]
    yc: Float[Tensor, "*B Nc Dy"]
    xt: Float[Tensor, "*B Nt Dx"]
    yt: Float[Tensor, "*B Nt Dy"]

    @classmethod
    def from_full_tensors(
        cls, x: Float[Tensor, "*B N Dx"], y: Float[Tensor, "*B N Dy"], Nc: int
    ):
        # when asked GPT about making compatible with multiple leading batch directions they would only suggest reshaping
        # think ... should work OK
        return cls(
            x=x,
            y=y,
            Nc=Nc,
            xc=x[..., :Nc, :],
            yc=y[..., :Nc, :],
            xt=x[..., Nc:, :],
            yt=y[..., Nc:, :],
        )

    def all_tensor_values_to_cuda(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                self.__dict__[k] = v.cuda()


@dataclass
class Checkpoint(AttrDict):
    """
    Checkpoint for saving/loading model state.
    """

    model_sd: StateDict
    optimizer_sd: StateDict
    scheduler_sd: StateDict
    logfilename: str
    step: int


@dataclass
class NPOutputs:
    """
    Outputs from forward pass of NP variant.
    (flexible for different NP variants)
    """

    tar_ll: Float[Tensor, "*B"]
