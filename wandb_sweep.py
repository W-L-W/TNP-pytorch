import logging

from typing import Callable, Optional


import wandb

from research_scaffold.util import recursive_dict_update
from gp_refac_min import gp_main


log = logging.getLogger(__name__)


function_map: dict[str, Callable] = {
    f.__name__: f
    for f in [
        gp_main,
    ]
}


def wandb_sweep(
    wandb_project: str,
    function_to_call: str,
    base_kwargs: dict,
    count: Optional[int] = None,
    sweep_configuration: Optional[dict] = None,
    sweep_id: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_entity: Optional[str] = None,
):

    def sweep_fn():
        wandb.init(group=wandb_group)
        kwargs = recursive_dict_update(base_kwargs, dict(wandb.config))
        function_map[function_to_call](**kwargs)

    if sweep_id is None:
        assert sweep_configuration is not None
        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project=wandb_project, entity=wandb_entity
        )
    else:
        assert sweep_configuration is None

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_fn,
        count=count,
        entity=wandb_entity,
        project=wandb_project,
    )
