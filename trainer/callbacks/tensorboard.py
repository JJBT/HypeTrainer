import os
from trainer.callbacks.callback import Callback
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf


class TensorBoardCallback(Callback):
    def __init__(self, frequency, add_weights=False, add_grads=False):
        super().__init__(frequency=frequency, before=True, after=True)
        self.log_dir = os.getcwd()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.add_weights = add_weights
        self.add_grads = add_grads

    def before_run(self, trainer):
        cfg = OmegaConf.to_yaml(trainer.cfg)
        cfg = cfg.replace('\n', '  \n')
        self.writer.add_text('cfg', cfg)
        description = trainer.cfg.description
        if description:
            self.writer.add_text('description', description)

    def after_run(self, trainer):
        self.writer.close()

    def add_validation_metrics(self, trainer):
        metrics = trainer.state.validation_metrics
        for name, metric in metrics.items():
            self.writer.add_scalar(name, metric, trainer.state.step)

    def add_weights_histogram(self, trainer):
        for name, param in trainer.model.named_parameters():
            if 'bn' not in name:
                self.writer.add_histogram(name, param, trainer.state.step)

    def add_grads_histogram(self, trainer):
        for name, param in trainer.model.named_parameters():
            if 'bn' not in name and param.requires_grad:
                self.writer.add_histogram(name + '_grad', param.grad, trainer.state.step)

    def __call__(self, trainer):
        for name, loss in trainer.state.last_train_loss.items():
            self.writer.add_scalar(f'trn/{name}', loss, trainer.state.step)

        self.writer.add_scalar('lr', trainer.optimizer.param_groups[0]['lr'], trainer.state.step)

        if self.add_weights:
            self.add_weights_histogram(trainer)

        if self.add_grads:
            self.add_grads_histogram(trainer)
