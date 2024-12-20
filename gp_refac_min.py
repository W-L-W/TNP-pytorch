"""Minimal refactoring of TNP repo code to be compatible with research scaffold"""

# standard library
import os
import os.path as osp
import argparse
import time
from copy import deepcopy
from typing import Literal
from dataclasses import dataclass

# third-party
import wandb
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from tqdm import tqdm

# project
from data.gp import *
from utils.misc import load_module, AttrDict, navigate_to_tnp_code_dir, StrKeyDict, CNP
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage
from utils.classes import Batch
from models.tnpa import TNPA


@dataclass
class GPExperimentArguments:
    """Adapt from original gp.py argument parsing code"""

    mode: Literal["train", "eval", "plot"]
    expid: str
    resume: bool = False

    # Data
    max_num_points: int = 50

    # Model
    model_name: str = "tnpa"
    model_kwargs: StrKeyDict = None

    # Train
    train_seed: int = 0
    train_batch_size: int = 16
    train_num_samples: int = 4
    train_num_bs: int = 10
    lr: float = 5e-4
    num_steps: int = 100000
    print_freq: int = 200
    eval_freq: int = 5000
    save_freq: int = 1000

    # Eval
    eval_seed: int = 0
    eval_num_batches: int = 3000
    eval_batch_size: int = 16
    eval_num_samples: int = 50
    eval_logfile: str = None

    # Plot
    plot_seed: int = 0
    plot_batch_size: int = 16
    plot_num_samples: int = 30
    plot_num_ctx: int = 30
    plot_num_tar: int = 10
    start_time: str = None
    plot_mode: Literal["original", "lennie"] = "original"

    # OOD settings
    eval_kernel: str = "rbf"
    t_noise: float = None

    @property
    def root(self) -> str:
        return osp.join(results_path, "gp", self.model_name, self.expid)

    def construct_model(self) -> CNP:
        model_cls = getattr(
            load_module(f"models/{self.model_name}.py"), self.model_name.upper()
        )
        return model_cls(**self.model_kwargs)


def gp_main(**exp_kwargs):
    args = GPExperimentArguments(**exp_kwargs)
    model = args.construct_model()
    model.cuda()

    if args.mode == "train":
        train(model, args)
    elif args.mode == "eval":
        eval(model, args, step=0)
    elif args.mode == "plot":
        plot(model, args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def train(model: CNP, args: GPExperimentArguments):
    if osp.exists(args.root + "/ckpt.tar"):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, "args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print("generating evaluation sets...")
        gen_evalset(args)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)

    sampler = GPSampler(RBFKernel())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )

    if args.resume:
        ckpt = AttrDict.load_torch(os.path.join(args.root, "ckpt.tar"))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(
            args.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log'
        )
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.model_name}-{args.expid}")
        logger.info(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n"
        )

    for step in range(start_step, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device="cuda",
        )

        if args.model_name in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            outs = model(batch, num_samples=args.train_num_samples)
        else:
            outs = model(batch)

        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({"train_loss": outs.loss.item()}, step=step)

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f"{args.model_name}:{args.expid} step {step} "
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(model, args, step=step)
                logger.info(line + "\n")

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            ckpt.save_torch(os.path.join(args.root, "ckpt.tar"))

    args.mode = "eval"
    eval(model, args, step=step)


def get_eval_path(args):
    path = osp.join(evalsets_path, "gp")
    filename = f"{args.eval_kernel}-seed{args.eval_seed}"
    if args.t_noise is not None:
        filename += f"_{args.t_noise}"
    filename += ".tar"
    return path, filename


def gen_evalset(args):
    if args.eval_kernel == "rbf":
        kernel = RBFKernel()
    elif args.eval_kernel == "matern":
        kernel = Matern52Kernel()
    elif args.eval_kernel == "periodic":
        kernel = PeriodicKernel()
    else:
        raise ValueError(f"Invalid kernel {args.eval_kernel}")
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(
            sampler.sample(
                batch_size=args.eval_batch_size,
                max_num_points=args.max_num_points,
                device="cuda",
            )
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))


def eval(model: CNP, args: GPExperimentArguments, *, step: int):
    # eval a trained model on log-likelihood
    if args.mode == "eval":
        ckpt = AttrDict.load_torch(
            os.path.join(args.root, "ckpt.tar"), map_location="cuda"
        )
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f"eval_{args.eval_kernel}"
            if args.t_noise is not None:
                eval_logfile += f"_tn_{args.t_noise}"
            eval_logfile += ".log"
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode="w")
    else:
        logger = None

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print("generating evaluation sets...")
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            batch.all_tensor_values_to_cuda()
            if args.model_name in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f"{args.model_name}:{args.expid} {args.eval_kernel} "
    if args.t_noise is not None:
        line += f"tn {args.t_noise} "
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    # Lennie: log to wandb!
    wandb.log(ravg.mean_dict(), step=step)

    return line


def eval_all_metrics(args, model):
    # eval a trained model on log-likelihood, rsme, calibration, and sharpness
    ckpt = AttrDict.load_torch(os.path.join(args.root, "ckpt.tar"), map_location="cuda")
    model.load_state_dict(ckpt.model)
    if args.eval_logfile is None:
        eval_logfile = f"eval_{args.eval_kernel}"
        if args.t_noise is not None:
            eval_logfile += f"_tn_{args.t_noise}"
        eval_logfile += f"_all_metrics"
        eval_logfile += ".log"
    else:
        eval_logfile = args.eval_logfile
    filename = os.path.join(args.root, eval_logfile)
    logger = get_logger(filename, mode="w")

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print("generating evaluation sets...")
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval_all_metrics":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        ravgs = [RunningAverage() for _ in range(4)]  # 4 types of metrics
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            if args.model_name in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                outs = model.predict(
                    batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples
                )
                ll = model(batch, num_samples=args.eval_num_samples)
            elif args.model_name in ["tnpa", "tnpnd"]:
                outs = model.predict(
                    batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples
                )
                ll = model(batch)
            else:
                outs = model.predict(batch.xc, batch.yc, batch.xt)
                ll = model(batch)

            mean, std = outs.loc, outs.scale

            # shape: (num_samples, 1, num_points, 1)
            if mean.dim() == 4:
                # variance of samples (Law of Total Variance) - var(X) = E[var(X|Y)] + var(E[X|Y])
                # E[var(X|Y)] : average variability within each samples
                # var(E[X|Y]) : variability between samples
                var = (
                    std.pow(2).mean(dim=0)
                    + mean.pow(2).mean(dim=0)
                    - mean.mean(dim=0).pow(2)
                )
                std = var.sqrt().squeeze(0)
                # mean of samples (Law of Total Expectations) - E[E[X|Y]] = E[X]
                mean = mean.mean(dim=0).squeeze(0)

            mean, std = (
                mean.squeeze().cpu().numpy().flatten(),
                std.squeeze().cpu().numpy().flatten(),
            )
            yt = batch.yt.squeeze().cpu().numpy().flatten()

            acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
            calibration = uct.metrics.get_all_average_calibration(
                mean, std, yt, num_bins=100, verbose=False
            )
            sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
            scoring_rule = {"tar_ll": ll.tar_ll.item()}

            batch_metrics = [acc, calibration, sharpness, scoring_rule]
            for i in range(len(batch_metrics)):
                ravg, batch_metric = ravgs[i], batch_metrics[i]
                for k in batch_metric.keys():
                    ravg.update(k, batch_metric[k])

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f"{args.model_name}:{args.expid} {args.eval_kernel} "
    if args.t_noise is not None:
        line += f"tn {args.t_noise} "

    line += "\n"

    for ravg in ravgs:
        line += ravg.info()
        line += "\n"

    if logger is not None:
        logger.info(line)

    return line


def flexible_predict(model: TNPA, xc, yc, xt, xp, num_samples=10):
    """Make AR predictions on the targets xt then roll out (parallel) input-wise predictions on
    the grid xp (of x-Plotting locations)"""
    from utils.misc import stack

    # TODO: debug this code!

    # lifting helper functions from tnpa.py
    batch_size = xc.shape[0]
    num_target = xt.shape[1]

    def squeeze(x):
        return x.view(-1, x.shape[-2], x.shape[-1])

    def unsqueeze(x):
        return x.view(num_samples, batch_size, x.shape[-2], x.shape[-1])

    # first make AR predictions with return sample mode
    # only implemented for tnpa for now
    yt_samples = model.predict(xc, yc, xt, num_samples=num_samples, return_samples=True)
    # dimension (num_samples, batch_size, num_target, dim_y)
    # then update batches to use this information
    # get xc2 by combining xc and xt then stacking by number of samples
    xc2 = stack(torch.cat((xc, xt), dim=-2), num_samples)
    # get yc2 by stacking yc by number of samples
    yc2 = torch.cat(stack(yc, num_samples), yt_samples, dim=-2)
    # plot points are simply stacked versions of earlier points
    xt2 = stack(xt, num_samples)
    xp2 = stack(xp, num_samples)
    yt2 = torch.zeros((batch_size, num_target, yc.shape[2]), device="cuda")
    # now get predictions in parallel
    batch_stacked = AttrDict()
    batch_stacked.xc = squeeze(xc2)
    batch_stacked.yc = squeeze(yc2)
    batch_stacked.xt = squeeze(xt2)
    batch_stacked.yt = squeeze(yt2)

    z_target_stacked = model.encode(batch_stacked, autoreg=True)
    out = model.predictor(z_target_stacked)
    mean, std = torch.chunk(out, 2, dim=-1)
    std = torch.exp(std)
    mean, std = unsqueeze(mean), unsqueeze(std)

    # how best to move forward? probably have to use the squeeze unsqueeze trick as it seems that
    # the model won't work in batched mode out of the box
    return mean, std


# extra thoughts:
# - nice to implement Batch creation from xc, yc, xt, yt (not full tensors necessarily)
#


def plot(model: CNP, args: GPExperimentArguments):
    """Currently this generates autoregressively on the full plot-design-grid."""
    seed = args.plot_seed
    num_smp = args.plot_num_samples

    if args.mode == "plot":
        ckpt = AttrDict.load_torch(
            os.path.join(args.root, "ckpt.tar"), map_location="cuda"
        )
        model.load_state_dict(ckpt.model)
    model = model.cuda()

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    kernel = RBFKernel()
    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)

    xp = torch.linspace(-2, 2, 200).cuda()
    batch = sampler.sample(
        batch_size=args.plot_batch_size,
        num_ctx=args.plot_num_ctx,
        num_tar=args.plot_num_tar,
        device="cuda",
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Nc = batch.xc.size(1)
    Nt = batch.xt.size(1)

    model.eval()
    with torch.no_grad():
        if args.model_name in ["np", "anp", "bnp", "banp"]:
            outs = model(batch, num_smp, reduce_ll=False)
        else:
            outs = model(batch, reduce_ll=False)
        tar_loss = outs.tar_ll  # [Ns,B,Nt] ([B,Nt] for CNP)
        if args.model_name in ["cnp", "canp", "tnpd", "tnpa", "tnpnd"]:
            tar_loss = tar_loss.unsqueeze(0)  # [1,B,Nt]

        if args.plot_mode == "original":
            xt = xp[None, :, None].repeat(args.plot_batch_size, 1, 1)
            if args.model_name in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd"]:
                pred = model.predict(batch.xc, batch.yc, xt, num_samples=num_smp)
            else:
                pred = model.predict(batch.xc, batch.yc, xt)

        elif args.plot_mode == "lennie":
            # sample AR over the target points, then return input-wise predictions on grid
            pass  # TODO fit in flexible_predict(model, batch.xc, batch.yc, batch.xt, xp, num_smp)

        mu, sigma = pred.mean, pred.scale

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size // 4, 1)
        ncols = min(4, args.plot_batch_size)
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()
    else:
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(
                    tnp(xp),
                    tnp(mu[s][i]),
                    color="steelblue",
                    alpha=max(0.5 / args.plot_num_samples, 0.1),
                )
                ax.fill_between(
                    tnp(xp),
                    tnp(mu[s][i]) - tnp(sigma[s][i]),
                    tnp(mu[s][i]) + tnp(sigma[s][i]),
                    color="skyblue",
                    alpha=max(0.2 / args.plot_num_samples, 0.02),
                    linewidth=0.0,
                )
            ax.scatter(
                tnp(batch.xc[i]),
                tnp(batch.yc[i]),
                color="k",
                label=f"context {Nc}",
                zorder=mu.shape[0] + 1,
            )
            ax.scatter(
                tnp(batch.xt[i]),
                tnp(batch.yt[i]),
                color="orchid",
                label=f"target {Nt}",
                zorder=mu.shape[0] + 1,
            )
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu[i]), color="steelblue", alpha=0.5)
            ax.fill_between(
                tnp(xp),
                tnp(mu[i] - sigma[i]),
                tnp(mu[i] + sigma[i]),
                color="skyblue",
                alpha=0.2,
                linewidth=0.0,
            )
            ax.scatter(
                tnp(batch.xc[i]), tnp(batch.yc[i]), color="k", label=f"context {Nc}"
            )
            ax.scatter(
                tnp(batch.xt[i]), tnp(batch.yt[i]), color="orchid", label=f"target {Nt}"
            )
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")

    plt.suptitle(f"{args.expid}", y=0.995)
    plt.tight_layout()

    save_dir_1 = osp.join(
        args.root, f"plot_num{num_smp}-c{Nc}-t{Nt}-seed{seed}-{args.start_time}.pdf"
    )
    file_name = "-".join(
        [
            args.model_name,
            args.expid,
            f"plot_num{num_smp}",
            f"c{Nc}",
            f"t{Nt}",
            f"seed{seed}",
            f"{args.start_time}.pdf",
        ]
    )
    if args.expid is not None:
        save_dir_2 = osp.join(results_path, "gp", "plot", args.expid, file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot", args.expid)):
            os.makedirs(osp.join(results_path, "gp", "plot", args.expid))
    else:
        save_dir_2 = osp.join(results_path, "gp", "plot", file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot")):
            os.makedirs(osp.join(results_path, "gp", "plot"))
    plt.savefig(save_dir_1)
    plt.savefig(save_dir_2)
    print(f"Evaluation Plot saved at {save_dir_1}\n")
    print(f"Evaluation Plot saved at {save_dir_2}\n")


if __name__ == "__main__":
    gp_main()
