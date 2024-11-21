from dataclasses import dataclass, field
from typing import Any
from jaxtyping import Array, Float


@dataclass(frozen=True)
class Batch:
    """
    Batch of (x,y) pairs for context and target points.
    Note have identity $N = Nc + Nt$
    """

    x: Float[Array, "B, N, Dx"]
    y: Float[Array, "B, N, Dy"]
    Nc: int

    xc: Float[Array, "B, Nc, Dx"] = field(init=False)
    yc: Float[Array, "B, Nc, Dy"] = field(init=False)
    xt: Float[Array, "B, Nt, Dx"] = field(init=False)
    yt: Float[Array, "B, Nt, Dy"] = field(init=False)

    def __post_init__(self):
        self.xc = self.x[:, : self.Nc, :]
        self.yc = self.y[:, : self.Nc, :]
        self.xt = self.x[:, self.Nc :, :]
        self.yt = self.y[:, self.Nc :, :]

    def tensors_to_cuda(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Array):
                self.__dict__[k] = v.cuda()


@dataclass
class Checkpoint:
    """
    Checkpoint for saving/loading model state.
    """

    model: Any
    optimizer: Any
    scheduler: any
    logfilename: str
    step: int
    # TODO improve type hints and finish implementing


@dataclass
class NPOutputs:
    """
    Outputs from forward pass of NP variant.
    (flexible for different NP variants)
    """

    tar_ll: Float[Array, "B"]
