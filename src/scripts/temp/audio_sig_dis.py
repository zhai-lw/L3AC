import numpy as numpy
import plotly.graph_objects as go

import utils
import xtract.data
from scripts import FIG_PATH

BIN_NUM = 9999


def init_dataset():
    ds_path = utils.file.DATA_PATH / "dataset/common_voice_24k/eval"
    # ds_path = utils.file.DATA_PATH / "dataset/FSD50K_44k/eval"
    # ds_path = utils.file.DATA_PATH / "dataset/libri_speech_FLAC/eval"
    # ds_path = utils.file.DATA_PATH / "dataset/mtg_low/eval"
    ds = xtract.data.XDataset.load_from_disk(ds_path)
    return ds.select_columns('audio').select(numpy.random.choice(len(ds), 200))


def weighted_mean_var(values: numpy.ndarray, weights: numpy.ndarray):
    mean = numpy.average(values, weights=weights)
    variance = numpy.average((values - mean) ** 2, weights=weights)
    return mean, variance


def get_audio_data(data_item):
    audio_data = data_item['audio']
    # return audio_data
    non_silence_audios = []
    spls = []
    for audio_slice in numpy.array_split(audio_data, len(audio_data) // 100):
        spl = 20 * numpy.log10(numpy.sqrt(numpy.power(audio_slice, 2).mean()) / 2) + 100
        spls.append(spl)
        if spl > 20:
            non_silence_audios.append(audio_slice)
    return numpy.concatenate(non_silence_audios)


def stat_his():
    dataset = init_dataset()
    total_histogram, ref_bins = numpy.zeros(BIN_NUM), numpy.arange(-1, 1, 2 / BIN_NUM)
    for data_item in utils.log.progressbar(dataset):
        audio_data = get_audio_data(data_item)
        histogram, bins = numpy.histogram(audio_data, bins=BIN_NUM, range=(-1, 1))
        assert numpy.allclose(bins[:-1], ref_bins)
        total_histogram += histogram

    p_dis = total_histogram / total_histogram.sum()
    ref_centers = ref_bins + 1 / BIN_NUM
    return p_dis, ref_centers


def check_mean_var(est_mean, est_var):
    print(f"est_mean: {est_mean}, est_var: {est_var}")
    dataset = init_dataset()
    welford = utils.data.WelFord()
    for data_item in utils.log.progressbar(dataset):
        welford.update(get_audio_data(data_item))
    print(f"actual_mean: {welford.mean}, actual_var: {welford.variance}")


def sim_compare():
    total_histogram, ref_bins = numpy.zeros(BIN_NUM), numpy.arange(-1, 1, 2 / BIN_NUM)
    for _ in utils.log.progressbar(range(100)):
        # rnd_s = numpy.random.randn(10000, 1000)
        rnd_s = numpy.random.rand(10000, 1000)
        # rnd_s = numpy.cos(rnd_s * numpy.pi)
        rnd_s = rnd_s * 2 - 1
        # rnd_s = numpy.exp(rnd_s)*numpy.sign(rnd_s)
        sim_s = numpy.mean(rnd_s, axis=1).clip(min=-1, max=1)
        histogram, bins = numpy.histogram(sim_s, bins=BIN_NUM, range=(-1, 1))
        assert numpy.allclose(bins[:-1], ref_bins)
        total_histogram += histogram
    p_dis = total_histogram / total_histogram.sum()
    ref_centers = ref_bins + 1 / BIN_NUM
    return p_dis, ref_centers


def dis_to_compare(ref_centers, p_mean, p_var):
    from scipy.stats import norm, laplace, irwinhall, gennorm, cauchy

    # for scale in (0.22, 0.25, 0.28):
    #     nor_line = norm.pdf(ref_centers, loc=p_mean, scale=scale)
    #     yield nor_line, f'normal_{scale}'

    for scale in numpy.linspace(0.08, 0.12, 3):
        lap_line = laplace.pdf(ref_centers, loc=p_mean, scale=scale)
        yield lap_line, f'laplace_{scale}'

    # for scale in numpy.linspace(0.05, 0.15, 3):
    #     lap_line = laplace.pdf(ref_centers, loc=p_mean, scale=scale)
    #     lap_exp_line = numpy.exp(lap_line-0.01)
    #     yield lap_exp_line, f'laplace_exp_{scale}'

    # for scale in (1, 0.1, 10):
    #     betas_line = irwinhall.pdf((ref_centers + 1) / 2 * scale, n=50, scale=1 / 50 * scale)
    #     yield betas_line, f'betas_{scale}'
    from scipy.special import gamma as gamma_func
    for gen_beta in (0.23, 0.22):
        for factor in (0.6, 0.5,):
            gen_alpha = (p_var * gamma_func(1 / gen_beta) / gamma_func(3 / gen_beta)) ** 0.5
            gen_line = gennorm.pdf(ref_centers*factor, beta=gen_beta, loc=p_mean, scale=gen_alpha)
            yield gen_line, f'gen_norm_{gen_beta:.2f}_f{factor}'
    for scale in (0.001, 0.0005):
        for factor in (0.1, 0.15, 0.2):
            cauchy_line = cauchy.pdf(ref_centers * factor, loc=p_mean, scale=scale)
            yield cauchy_line, f'cauchy_{scale}_f{factor}'


def main():
    p_dis, ref_centers = stat_his()
    p_mean, p_var = weighted_mean_var(ref_centers, weights=p_dis)
    # check_mean_var(p_mean, p_var)

    fig_dis = go.Figure()
    fig_dis.add_trace(go.Bar(x=ref_centers, y=p_dis * BIN_NUM, name='Audio'))

    # sim_dis, sim_centers = sim_compare()
    # sim_mean, sim_var = weighted_mean_var(sim_centers, weights=sim_dis)
    # print(f"sim_mean: {sim_mean}, sim_var: {sim_var}")
    # fig_dis.add_trace(go.Bar(x=sim_centers, y=sim_dis, name='SIM'))

    for line_y, line_name in dis_to_compare(ref_centers, p_mean, p_var):
        fig_dis.add_trace(go.Scatter(x=ref_centers, y=line_y, mode='lines', name=line_name))

    fig_dis.update_yaxes(type="log")
    fig_dis.show()
    fig_dis.write_html(FIG_PATH / "audio_signal_distribution.html")
    print("done")


def temp():
    import numpy as np
    from scipy.stats import irwinhall
    import matplotlib.pyplot as plt

    n = 500  # Number of uniform random variables
    bates_dist = irwinhall(n, scale=1 / n)

    # Generate some random variates
    samples = bates_dist.rvs(size=1000)

    # Plot the PDF
    x = np.linspace(0, 1, 100)
    pdf_values = bates_dist.pdf(x)

    plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Bates Samples')
    plt.plot(x, pdf_values, 'r-', lw=2, label='Bates PDF')
    plt.title(f'Bates Distribution (n={n})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Get mean and variance
    print(f"Mean of Bates distribution (n={n}): {bates_dist.mean()}")
    print(f"Variance of Bates distribution (n={n}): {bates_dist.var()}")


if __name__ == '__main__':
    # temp()
    main()
