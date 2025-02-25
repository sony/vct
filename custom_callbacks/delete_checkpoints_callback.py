import lightning as L
import wandb


class DeleteCheckpointsCallback(L.Callback):

    def __init__(self, path, every_n_iterations=1):
        super().__init__()
        self.every_n_iterations = every_n_iterations  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.path = path
        self.run = wandb.Api().run(self.path)

    def _delete_artifacts_without_alias(self):
        for artifact_version in self.run.logged_artifacts():
            # Keep only artifacts with alias "best" or "latest"
            if len(artifact_version.aliases) == 0:
                artifact_version.delete()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            if trainer.global_step % self.every_n_iterations == 0:
                self._delete_artifacts_without_alias()
