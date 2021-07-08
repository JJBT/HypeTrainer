from trainer.callbacks.callback import Callback


class ValidationCallback(Callback):
    def __init__(self, frequency):
        super().__init__(frequency=frequency, before=False, after=False)

    def __call__(self, trainer):
        self.computed_metrics = trainer.evaluate(dataloader=trainer.val_dataloader, metrics=trainer.metrics)
        for metric_name, metric_value in self.computed_metrics.items():
            trainer.state.add_validation_metric(name=f'val/{metric_name}', value=metric_value)

        trainer.state.log_validation()

        if 'TensorBoardCallback' in trainer.callbacks:
            tb_callback = trainer.callbacks['TensorBoardCallback']
            tb_callback.add_validation_metrics(trainer)
