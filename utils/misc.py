import os
from typing import Any, Dict
from importlib.machinery import SourceFileLoader
import math
import torch


def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline

    return load


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    # previously gave file not found so made filepath relative...
    par_path = os.path.dirname
    tnp_dir = par_path(par_path(os.path.realpath(__file__)))
    file_path = os.path.join(tnp_dir, filename)
    return SourceFileLoader(module_name, file_path).load_module()
    # <module "module_name" from "filename">
    #
    # ex.
    # <module "cnp" from "models/cnp.py">


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x, num_samples=None, dim=0):
    return x if num_samples is None else torch.stack([x] * num_samples, dim=dim)


def hrminsec(duration):
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{hours}hrs {mins}mins {secs}secs"


########
# Lennie
########

StrKeyDict = Dict[str, Any]
NestedTensorDict = StrKeyDict
# convention: this is a dictionary whose keys are all strings
# and values are either Tensors or NestedTensorDicts
# (but that is hard to type hint)


def move_nested_tensor_values_to_cuda(d: NestedTensorDict):
    """Move all tensor leaves to cuda in place."""
    for k, v in d.items():
        assert isinstance(
            k, str
        ), f"Unexpected key type for NestedTensorDict: {type(k)}"
        if isinstance(v, dict):
            move_nested_tensor_values_to_cuda(v)
        elif isinstance(v, torch.Tensor):
            d[k] = v.cuda()
        else:
            raise ValueError(f"Unexpected value type for NestedTensorDict: {type(v)}")


def navigate_to_tnp_code_dir():
    # previously gave file not found so made filepath relative...
    par_path = os.path.dirname
    tnp_dir = par_path(par_path(os.path.realpath(__file__)))
    os.chdir(tnp_dir)
    print("Navigated to tnp dir: ", os.getcwd())


class AttrDict:
    """Lightweight implementation of attribute dictionary (by Lennie).
    Class only supports attribute access and setting.
    However it has a dictionary property for when one wants a plain dictionary interface.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def dictionary(self):
        return self.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.dictionary)}"

    def items(self):
        return self.dictionary.items()

    def save_torch(self, f):
        """Save attr dict to torch file."""
        torch.save(self.dictionary, f)

    @classmethod
    def load_torch(
        cls, f, map_location=None, *, weights_only=True, mmap=None, **pickle_load_args
    ):
        """Load attr dict from torch file."""
        torch_dict = torch.load(
            f,
            map_location=map_location,
            weights_only=weights_only,
            mmap=mmap,
            **pickle_load_args,
        )
        return cls.__init__(**torch_dict)


from collections import UserDict

# class AttrDict(UserDict):
#     """Work around suggested at https://stackoverflow.com/questions/72361026/how-can-i-get-attrdict-module-working-in-python
#     (not confident this is full replacement)"""
#     def __getattr__(self, key):
#         return self.__getitem__(key)
#     def __setattr__(self, key, value):
#         if key == "data":
#             return super().__setattr__(key, value)
#         return self.__setitem__(key, value)


# class AttrDict(dict):
# '''Class that is like a dictionary with items usable like attributes.
# From https://python-forum.io/thread-15082.html
# #---------------------------------------------------------------
# # purpose       class that is a dictionary with items usable
# #               like attributes
# #
# # init usage    object = attrdict(dictionary)
# #               object = attrdict(dictionary,key=value...)
# #               object = attrdict(key=value...)
# #
# # attr usage    object.name
# #
# # dict usage    object[key]
# #
# # note          attribute usage is like string keys that are
# #               limited to what can be a valid identifier.
# #
# # thanks        nilamo@python-forum.io
# #---------------------------------------------------------------
# '''
# def __init__(self,*args,**opts):
#     arn = 0
#     for arg in args:
#         arn += 1
#         if isinstance(arg,(attrdict,dict)):
#             self.update(arg)
#         elif arg and isinstance(arg,(list,tuple)):
#             an = -1
#             for ar in arg:
#                 an += 1
#                 if isinstance(ar,(list,tuple)) and len(ar)==2:
#                     self[ar[0]] = ar[1]
#                 else:
#                     raise TypeError('not a 2-sequence at ['+str(an)+'] of argument '+str(arn))
#         else:
#             raise TypeError('argument '+str(arn)+' is not a sequence')
#     if opts:
#         if isinstance(opts,(attrdict,dict)):
#             self.update(opts)
#         else:
#             raise TypeError('options ('+repr(opts)+') is not a dictionary')
# def __getattr__(self, key):
#     return self[key]
# def __setattr__(self, key, value):
#     self[key] = value
