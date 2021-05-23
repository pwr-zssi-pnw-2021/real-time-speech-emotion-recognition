# %%
import pytorch_lightning as pl

from train.datamodules import TESSDatamodule
from train.models import TESSConvModel, TESSLinearModel

pl.seed_everything(42)

# Linear
# m = TESSLinearModel()
# dm = TESSDatamodule(256)

# Conv
m = TESSConvModel()
dm = TESSDatamodule(256, data_dim=2)

t = pl.Trainer(gpus=1, max_epochs=100)
t.fit(m, datamodule=dm)
# %%
