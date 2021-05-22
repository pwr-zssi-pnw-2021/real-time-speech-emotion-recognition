# %%
import pytorch_lightning as pl

from train.datamodules import TESSDatamodule
from train.models import TESSLinearModel

pl.seed_everything(42)

m = TESSLinearModel()
dm = TESSDatamodule(256)

t = pl.Trainer(gpus=1, max_epochs=100)
t.fit(m, datamodule=dm)
# %%
