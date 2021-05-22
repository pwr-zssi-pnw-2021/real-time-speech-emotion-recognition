# %%
from pathlib import Path

import IPython.display as ipd
import librosa
import librosa.display as libd
import matplotlib.pyplot as plt

# %%
# p = Path('../data/TESS/YAF_happy/YAF_bath_happy.wav')
p = Path('../data/AFEW_wav/Train/Fear/000852880.wav')
y, sr = librosa.core.load(p)
# %%
plt.plot(y)

feat = librosa.feature.mfcc(y)
libd.specshow(feat, x_axis='time')
# %%
feat.shape
# %%
sr
# %%
