import contextlib
import pathlib

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import signal

# requirements: pip install scipy seaborn
import seaborn


# cmap options: viridis, hot, gray, jet, bone, gray_r, rainbow, magma

def new_fig(layout='constrained', dpi=300, fig_size=None, return_ax=True) -> (Figure, Axes):
    fig = plt.figure(layout=layout, figsize=fig_size, dpi=dpi)
    if return_ax:
        return fig, fig.gca()
    else:
        return fig


class Draw:

    def __init__(self, legend=True, default_title=None, **ax_kwargs):
        self.legend = legend
        self.default_title = default_title
        self.ax_kwargs = ax_kwargs

    def __call__(self, plot_func):
        def wrapper(*data, title: str = self.default_title, ax2p: Axes = None,
                    show=True, out_path: pathlib.Path = None, **kwargs):
            with self.get_ax(ax2p, show=show, out_path=out_path) as ax2p:
                plot_func(*data, ax2p=ax2p, **kwargs)
                if title:
                    ax2p.set_title(title)
                if self.legend:
                    ax2p.legend(fontsize="large")

            return ax2p.figure

        wrapper.without_wrapper = plot_func
        return wrapper

    def get_ax(self, ax2p: Axes, show: bool, out_path: pathlib.Path):
        if ax2p:
            return contextlib.nullcontext(ax2p)
        else:
            return self.create_ax(show=show, out_path=out_path)

    @contextlib.contextmanager
    def create_ax(self, show: bool, out_path: pathlib.Path):
        fig, ax2p = new_fig(**self.ax_kwargs)

        yield ax2p

        if show:
            fig.show()
        if out_path is not None:
            fig.savefig(out_path, dpi=1000)


@contextlib.contextmanager
def subplots(rows_num=1, cols_num=1, show: bool = True, out_path: pathlib.Path = None, **kwargs):
    fig, axes = plt.subplots(nrows=rows_num, ncols=cols_num, layout='constrained', **kwargs)
    fig.set_size_inches(16, 10)
    yield axes
    if show:
        fig.show()
    if out_path is not None:
        fig.savefig(out_path, dpi=1000)


@Draw()
def line(x, ax2p: Axes):
    ax2p.plot(x)


@Draw()
def scatter(data, labels, ax2p: Axes):
    for label in set(labels):
        ax2p.scatter(data[labels == label, 0], data[labels == label, 1], label=label)


@Draw(projection='3d')
def complex_line_3d(array_data: numpy.ndarray, ax2p: Axes):
    x = range(len(array_data))
    y = array_data.real
    z = array_data.imag
    ax2p.plot(x, y, z)
    # ax.contour(x, y, z, zdir='x')
    # ax.contour(x, y, z, zdir='y')
    # ax.contour(x, y, z, zdir='z')


@Draw(default_title="heatmap", legend=False)
def heatmap(data, ax2p: Axes, color_map='rainbow'):
    # data = (data - data.min()) / (data.max() - data.min())
    seaborn.heatmap(data, ax=ax2p, cmap=color_map)


@Draw()
def pixels(data, ax2p: Axes, color_map='rainbow'):
    ax2p.imshow(data, cmap=color_map, aspect='auto')


@Draw(default_title="distribution", legend=False)
def distribution(data, ax2p: Axes):
    if isinstance(data, dict):
        palette = seaborn.color_palette(None, len(data))
        for (key, value), color in zip(data.items(), palette):
            seaborn.histplot(value.flatten(), label=key, kde=True, ax=ax2p, color=color)
    elif isinstance(data, list):
        palette = seaborn.color_palette(None, len(data))
        for value, color in zip(data, palette):
            seaborn.histplot(value.flatten(), kde=True, ax=ax2p, color=color)
    else:
        seaborn.histplot(data.flatten(), kde=True, ax=ax2p)


@Draw(default_title="image", legend=False)
def image(data: numpy.ndarray, ax2p: Axes):
    if data.ndim == 3:
        if data.shape[0] == 1 or data.shape[0] == 3:
            data = data.transpose((1, 2, 0))
    ax2p.imshow(data)


@Draw(default_title="Spectrogram")
def spectrogram(spectrogram_data: numpy.ndarray, ax2p: Axes, time_length=1.0, freq_length=8000,
                sample_frequencies=None, segment_times=None, color_map='jet', **kwargs):
    # spectrogram_data.shape = (sample_frequencies, segment_times)
    if sample_frequencies is None:
        sample_frequencies = numpy.linspace(0, freq_length, spectrogram_data.shape[0])
    if segment_times is None:
        segment_times = numpy.linspace(0, time_length, spectrogram_data.shape[1])
    pcolor = ax2p.pcolormesh(segment_times, sample_frequencies, spectrogram_data,
                             cmap=color_map, **kwargs)
    ax2p.set_ylabel('Frequency')
    ax2p.set_xlabel('Time')
    ax2p.figure.colorbar(pcolor, ax=ax2p)


@Draw(default_title="Spectrogram", legend=False)
def audio_spectrogram(audio_data, ax2p: Axes, log_mod=True,
                      fs=16000, nfft=1024, window='hann', nperseg=1024, noverlap=768,
                      v_max=None, v_min=None):
    """
    audio_data: audio data (length,) dtype=float
    """

    f, t, sxx = signal.spectrogram(audio_data, fs=fs, nfft=nfft, window=window, nperseg=nperseg,
                                   noverlap=noverlap, scaling="spectrum")
    if log_mod:
        sxx = 10 * (numpy.log10(sxx + 1e-8))
    spectrogram.without_wrapper(sxx, ax2p=ax2p,
                                sample_frequencies=f, segment_times=t,
                                vmax=v_max, vmin=v_min)


@Draw(default_title="data_matrix", legend=False)
def data_matrix(data: numpy.ndarray, ax2p: Axes, color_map='rainbow',
                x_ticks: list['str'] = None, y_ticks=None, x_name: str = None, y_name: str = None,
                **heatmap_kwargs):
    if x_ticks is None:
        x_ticks = range(data.shape[1])
    if y_ticks is None:
        y_ticks = range(data.shape[0])
    seaborn.heatmap(data, ax=ax2p, annot=True, cmap=color_map,
                    xticklabels=x_ticks, yticklabels=y_ticks, **heatmap_kwargs)
    ax2p.set_xticklabels(ax2p.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax2p.set_yticklabels(ax2p.get_yticklabels(), rotation="horizontal")
    if x_name:
        ax2p.set_xlabel(x_name)
    if y_name:
        ax2p.set_ylabel(y_name)


@Draw(default_title="confusion_matrix", legend=False)
def confusion_matrix(matrix_data: numpy.ndarray, ax2p: Axes, label_names: list = None, output_probability=False,
                     font_size=6, **matrix_kwargs):
    """
    matrix_data[0, :]: predict_cls_list
    matrix_data[:, 0]: target_cls_list
    """
    x_name, y_name = 'predict', 'target'
    num_per_cls = matrix_data.sum(axis=1)
    x_ticks = label_names
    y_ticks = [f"{name}-{num}" for name, num in zip(label_names, num_per_cls)]
    fmt = ""
    if output_probability:
        matrix_data = matrix_data / (num_per_cls[:, numpy.newaxis] + 1e-8)
        acc_per_cls = numpy.diag(matrix_data)
        mean_acc_per_cls = acc_per_cls.mean() * 100
        all_acc = numpy.sum(acc_per_cls * num_per_cls) / numpy.sum(num_per_cls) * 100
        matrix_data = numpy.concatenate([acc_per_cls[:, numpy.newaxis], matrix_data], axis=1)
        matrix_data *= 100
        x_ticks = [f'acc({all_acc:2.1f} | pc:{mean_acc_per_cls:2.1f})'] + x_ticks
        fmt = "2.1f"
    data_matrix.without_wrapper(matrix_data, ax2p=ax2p, x_ticks=x_ticks, y_ticks=y_ticks,
                                x_name=x_name, y_name=y_name,
                                fmt=fmt, annot_kws={"size": font_size}, **matrix_kwargs)
