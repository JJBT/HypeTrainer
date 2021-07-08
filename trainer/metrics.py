import torch
from trainer.metric_transforms import transforms_dict
from sklearn.metrics import precision_score, recall_score
import numpy as np


class Metric:
    def __init__(self, name: str, default_value=None, target_transform=None, prediction_transform=None):
        self.name = name.replace(' ', '_')
        self.default_value = default_value
        self.target_transform = target_transform if target_transform else \
            transforms_dict.get(f'{self.name}_target', lambda x: x)
        self.prediction_transform = prediction_transform if prediction_transform else \
            transforms_dict.get(f'{self.name}_prediction', lambda x: x)

    def prepare(self, y: torch.Tensor, y_pred: torch.Tensor):
        y = self.target_transform(y)
        y_pred = self.prediction_transform(y_pred)

        if isinstance(y, torch.Tensor):
            y = y.detach()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach()

        return y, y_pred

    def step(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class Accuracy(Metric):
    def __init__(self):
        super().__init__("accuracy", default_value=0)
        self.total_correct = 0
        self.total = 0

    def step(self, y, y_pred):
        y, y_pred = self.prepare(y, y_pred)
        correct = torch.eq(y, y_pred)

        self.total_correct += torch.sum(correct).item()
        self.total += correct.shape[0]

    def compute(self):
        return self.total_correct / self.total

    def reset(self):
        self.total_correct = 0
        self.total = 0


class Recall(Metric):
    def __init__(self, average=None, target_transform=None, prediction_transform=None):
        super().__init__('recall', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.predicions = []
        self.targets = []
        self.average = average

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        # TODO
        y, y_pred = self.prepare(y, y_pred)
        self.targets.extend(y.tolist())
        self.predicions.extend(y_pred.tolist())

    def compute(self):
        result = recall_score(self.targets, self.predicions, average=self.average)
        if self.average:
            return result
        else:
            return {f'{i}_recall': result[i] for i in range(result.shape[0])}

    def reset(self):
        self.predicions = []
        self.targets = []


class Precision(Metric):
    def __init__(self, average=None, target_transform=None, prediction_transform=None):
        super().__init__('precision', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.predicions = []
        self.targets = []
        self.average = average

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        # TODO
        y, y_pred = self.prepare(y, y_pred)
        self.targets.extend(y.tolist())
        self.predicions.extend(y_pred.tolist())

    def compute(self):
        result = precision_score(self.targets, self.predicions, average=self.average)
        if self.average:
            return result
        else:
            return {f'{i}_precision': result[i] for i in range(result.shape[0])}

    def reset(self):
        self.predicions = []
        self.targets = []


class ConfusionMatrix(Metric):
    def __init__(self, num_classes, target_transform=None, prediction_transform=None):
        super().__init__('conf_matrix', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        y, y_pred = self.prepare(y, y_pred)
        y, y_pred = y.tolist(), y_pred.tolist()

        for t, p in zip(y, y_pred):
            self.matrix[int(t)][int(p)] += 1

    def compute(self):
        return self.matrix

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))
