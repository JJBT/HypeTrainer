import torch


transforms_dict = {
    'accuracy_prediction': lambda x: torch.argmax(x, dim=1),
    'accuracy_target': lambda x: torch.argmax(x, dim=1),
    'recall_prediction': lambda x: torch.argmax(x, dim=1),
    'recall_target': lambda x: torch.argmax(x, dim=1),
    'precision_prediction': lambda x: torch.argmax(x, dim=1),
    'precision_target': lambda x: torch.argmax(x, dim=1),
    'conf_matrix_prediction': lambda x: torch.argmax(x, dim=1),
    'conf_matrix_target': lambda x: torch.argmax(x, dim=1),
}
