import pytorch_lightning as pl

class EarlyStoppingLR(pl.callbacks.base.Callback):
    def __init__(self, min_lr):
        self.min_lr = min_lr
    
    def on_epoch_start(self, trainer, pl_module):
        current_lr = trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        if self.min_lr > current_lr:
            trainer.should_stop = True