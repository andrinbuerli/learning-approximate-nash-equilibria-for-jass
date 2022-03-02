import os
import wandb

from lib.log.base_logger import BaseLogger


class WandbLogger(BaseLogger):

    def __init__(
            self,
            wandb_project_name: str,
            group_name: str,
            api_key: str,
            config: dict,
            entity: str = None,
            run_name: str = None):
        super().__init__(config=config)

        os.environ["WANDB_API_KEY"] = api_key
        wandb.login()

        self.run: wandb = wandb.init(project=wandb_project_name, entity=entity, name=run_name, group=group_name)
        wandb.config.update(self.config)

    def log(self, data: dict):
        wandb.log(data)

    def dispose(self):
        self.run.finish()
