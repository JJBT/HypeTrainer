import torch
from trainer.utils.utils import object_from_dict
from torch.utils.data import DataLoader


class Factory:
    def __init__(self, cfg):
        self.cfg = cfg

    def create_model(self):
        model = object_from_dict(self.cfg.model)
        return model

    def create_optimizer(self, model: torch.nn.Module):
        optimizer = object_from_dict(self.cfg.optimizer, params=filter(lambda x: x.requires_grad, model.parameters()))
        return optimizer

    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        scheduler = object_from_dict(self.cfg.scheduler, optimizer=optimizer)
        return scheduler

    def create_loss(self):
        loss = object_from_dict(self.cfg.loss)
        return loss

    def create_train_dataloader(self):
        dataset = self.create_dataset(self.cfg.data.train_dataset)
        train_dataloader = self.create_dataloader(self.cfg.bs, dataset)
        return train_dataloader

    def create_val_dataloader(self):
        dataset = self.create_dataset(self.cfg.data.validation_dataset)
        val_dataloader = self.create_dataloader(self.cfg.bs, dataset)
        return val_dataloader

    def create_dataset(self, cfg):
        augmentations = self.create_augmentations(cfg.augmentations)
        dataset = object_from_dict(cfg, transforms=augmentations, ignore_keys=['augmentations'])
        return dataset

    def create_dataloader(self, bs, dataset):
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        return dataloader

    def create_metrics(self):
        metrics = []
        for metric in self.cfg.metrics:
            metric_obj = object_from_dict(metric)
            metrics.append(metric_obj)

        return metrics

    def create_callbacks(self, trainer):
        for hook in self.cfg.hooks:
            hook_obj = object_from_dict(hook)
            trainer.register_callback(hook_obj)

    def create_augmentations(self, cfg):
        augmentations = []
        for augm in cfg.augmentations:
            augmentations.append(object_from_dict(augm))

        compose = object_from_dict(cfg.compose, transforms=augmentations)
        return compose
