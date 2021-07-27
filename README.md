# HypeTrainer (v0.1)

HypeTrainer is a trainer for PyTorch models.

## Installation

```bash
git clone https://github.com/JJBT/HypeTrainer.git
cd HypeTrainer
python3 setup.py install --user
```

## Usage

```python
from trainer.factory import Factory
from trainer.trainer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf

def run_train(cfg):
    factory = Factory(cfg)
    trainer = Trainer(cfg, factory)
    trainer.run_train()

# Config format in examples (or you can override trainer.factory.Factory)
@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    run_train(cfg)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

