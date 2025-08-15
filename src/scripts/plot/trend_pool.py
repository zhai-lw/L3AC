import math

import librosa
import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib.ticker import FormatStrFormatter

from scripts.plot import PLOT_DIR

plt.rcParams['font.family'] = 'Times New Roman'

FONT_SIZE = 15
plt.rcParams['font.size'] = FONT_SIZE

# paper fig

sample_rate = 16000
ts, _ = librosa.load(PLOT_DIR / "raw_data/Middle_C.wav", sr=sample_rate, mono=True)
print(ts.shape)
ts = ts[:sample_rate * 1]
# %%
from l3ac.tconv.base import TrendPool

t_pool = TrendPool(kernel_size=50)
tts = t_pool(torch.tensor(ts)[None, None, :])[0, 0, :len(ts)]
# %%
f_min, f_max = int(0.5 * sample_rate), int(0.6 * sample_rate)
target_min, target_max = f_min, f_min + 1000
target_x = f_min + 20
# %%
fig, axes = plt.subplots(4, 1, figsize=(6.8, 6.1))
fig.tight_layout()
# %%
yf = numpy.fft.rfft(ts, norm='forward')
yf_abs = numpy.abs(yf)
axes[0].plot(yf_abs, color='#826ba2')
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Magnitude")
axes[0].yaxis.set_label_coords(x=-0.007*FONT_SIZE, y=0.5)
axes[0].set_title(" (a)", loc="right", x=1., y=0.4, rotation=0,
                  ha="left", va="center", fontsize=FONT_SIZE + 6)

# axes[0].specgram(ts)

# %%
xs = numpy.arange(0, sample_rate, 1) / sample_rate * 1000  # ms
axes[1].plot(xs, ts, color='#298eb1')  # 298eb1 #298eb1
axes[1].plot(xs, tts, color='#c22f2f')
box_height = ts.max() * 0.8
ymin, ymax = axes[1].get_ybound()
axes[1].plot(xs[target_min: target_max], numpy.zeros(target_max - target_min) + box_height,
             color='#449545', linestyle='dashed')
axes[1].plot(xs[target_min: target_max], numpy.zeros(target_max - target_min) - box_height,
             color='#449545', linestyle='dashed')
axes[1].axvline(x=xs[target_min], ymax=0.5 + box_height / ymax / 2, ymin=0.5 + box_height / ymin / 2,
                color='#449545', linestyle='dashed')
axes[1].axvline(x=xs[target_max], ymax=0.5 + box_height / ymax / 2, ymin=0.5 + box_height / ymin / 2,
                color='#449545', linestyle='dashed')
# axes[1].xaxis.set_major_formatter(FormatStrFormatter('%dms'))
axes[1].set_yticks([-0.09, 0.0, 0.09, ])
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Magnitude')
axes[1].yaxis.set_label_coords(x=-0.007*FONT_SIZE, y=0.5)
axes[1].set_title(" (b)", loc="right", x=1., y=0.4, rotation=0,
                  ha="left", va="center", fontsize=FONT_SIZE + 6)

# %%
axes[2].plot(xs[f_min: f_max], ts[f_min: f_max], color='#298eb1')
# axes[2].xaxis.set_major_formatter(FormatStrFormatter('%dms'))
axes[2].set_yticks([-0.05, 0.0, 0.05, ])
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Magnitude')
axes[2].yaxis.set_label_coords(x=-0.007*FONT_SIZE, y=0.5)
axes[2].set_title(" (c)", loc="right", x=1., y=0.4, rotation=0,
                  ha="left", va="center", fontsize=FONT_SIZE + 6)

# %%
axes[3].plot(xs[f_min: f_max], numpy.abs(ts[f_min: f_max]), color='#298eb1')
axes[3].plot(xs[f_min: f_max], tts[f_min: f_max], color='#c22f2f')
# axes[3].xaxis.set_major_formatter(FormatStrFormatter('%dms'))
axes[3].set_ylim(ymin=-0.01)
axes[3].set_yticks([0.0, 0.05, ])
axes[3].set_xlabel('Time (ms)')
axes[3].set_ylabel('Abs Magnitude')
axes[3].yaxis.set_label_coords(x=-0.007*FONT_SIZE, y=0.5)
axes[3].set_title(" (d)", loc="right", x=1., y=0.4, rotation=0,
                  ha="left", va="center", fontsize=FONT_SIZE + 6)

# %%
plt.savefig(PLOT_DIR / "trend_pool.pdf", format="pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()

print("done")
