import math

import matplotlib.pyplot as plt

from scripts.plot import PLOT_DIR

# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'Times New Roman'

FONT_SIZE = 20
plt.rcParams['font.size'] = FONT_SIZE

# -------------------------
# 示例数据
# -------------------------
# x = MAC，
# y = PESQ,
# s = 作为bubble的大小，与 bitrate 相关,
# label = 气泡旁边的文字标注
chart_data = {
    'DAC': {
        'data': [
            (556.0, 1.08, 0.58, 0.5, '0.5kbps'),
            (556.01, 1.18, 0.70, 1.0, '1kbps'),
            (556.02, 1.31, 0.76, 1.5, '1.5kbps'),
            (556.05, 1.92, 0.87, 3, '3kbps'),
        ],
        'color': '#D41F2F',
        'marker': 'o',
    },
    'Encodec': {
        'data': [
            (55.95, 1.50, 0.80, 1.5, '1.5kbps'),
            (55.95, 1.91, 0.86, 3, '3kbps'),
        ],
        'color': '#6D3789',
        'marker': 'o',
    },
    'WavTokenizer': {
        'data': [
            (34.26, 1.51, 0.75, 0.48, '0.48kbps'),
            (64.17, 1.58, 0.77, 0.9, '0.9kbps'),
        ],
        'color': '#7ABE52',
        'marker': 'o',
    },
    'TAAE': {
        'data': [
            (374.89, 1.53, 0.76, 0.39, '0.39kbps'),
            (374.89, 1.64, 0.79, 0.7, '0.7kbps'),
        ],
        'color': '#33B0D3',
        'marker': 'o',
    },
    'UniCodec': {
        'data': [
            (71.22, 1.91, 0.84, 1.05, '1.05kbps'),
        ],
        'color': '#EEB930',
        'marker': 'o',
    },
    'Ours': {
        'data': [
            # (MAC, PESQ, STOI, bps, 'text'),
            (1.55, 1.68, 0.82, 0.75, '0.75kbps'),
            (2.03, 1.77, 0.84, 1.0, '1.0kbps'),
            (2.37, 1.91, 0.86, 1.5, '1.5kbps'),
            (1.64, 2.31, 0.90, 3, '3kbps'),
        ],
        'color': 'blue',
        'marker': '*',
    },
}

# -------------------------
plt.figure(figsize=(8, 6))
ax = plt.gca()

# plt.axhline(y=10.0, color='orange', linestyle='--', label='UTMOS=4.0 ref')

UNIT_SIZE = FONT_SIZE * 10

METRICS = {
    'BPS': dict(range=(0., 3.2), label='Bitrate (bps)'),
    'MACs': dict(range=(1, 1000), label='Complexity (GMACs)'),
    'STOI': dict(range=(0.55, 0.95), label='STOI'),
    'PESQ': dict(range=(1, 3), label='PESQ'),
}
NX, NY, NS = 'BPS', 'STOI', 'MACs'
X_RANGE, Y_RANGE = METRICS[NX]['range'], METRICS[NY]['range']
X_MIDDLE, Y_MIDDLE = (X_RANGE[0] + X_RANGE[1]) / 2, (Y_RANGE[0] + Y_RANGE[1]) / 2
X_UNIT, Y_UNIT = (X_RANGE[1] - X_RANGE[0]) / 100, (Y_RANGE[1] - Y_RANGE[0]) / 100
# 依次绘制每种方法的数据点（气泡）
texts = []
for method, m_data in chart_data.items():
    xs, ys = [], []
    for macs, pesq, stoi, bps, label in m_data['data']:
        x, y, size = bps, stoi, math.log10(macs) * 14
        ax.scatter(x, y, s=size * UNIT_SIZE, c=m_data['color'], alpha=0.6, marker=m_data['marker'])
        xs.append(x)
        ys.append(y)

        # text_x = x + 0.5 * X_UNIT * (8 + math.sqrt(size)) * 1  # (1 if x < X_MIDDLE else -1)
        # text_y = y + 0.5 * Y_UNIT * (8 + math.sqrt(size)) * (-1)
        # texts.append(ax.text(x, y, label, ha='center', va='bottom', fontsize=14, color='black'))
    ax.plot(xs, ys, c=m_data['color'], linestyle='dashed')
    # ax.plot(xs, ys, c=m_data['color'], linestyle='none')
    # 图例
    ax.scatter(-10, -10, s=UNIT_SIZE, c=m_data['color'], alpha=0.6, marker=m_data['marker'], label=method)

# adjust_text(texts)

# -------------------------
# 坐标轴与标题
# -------------------------
plt.xlabel(METRICS[NX]['label'], fontsize=FONT_SIZE)
plt.ylabel(METRICS[NY]['label'], fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlim(*X_RANGE)
plt.ylim(*Y_RANGE)
# plt.xscale("log")

# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=FONT_SIZE)
# plt.legend(loc='upper left', fontsize=FONT_SIZE)

# -------------------------
# 显示或保存
# -------------------------
# plt.title("Bubble Chart", fontsize=20)
plt.tight_layout()
plt.savefig(PLOT_DIR / "bubble_chart.pdf", format="pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()

print("Bubble Chart")
