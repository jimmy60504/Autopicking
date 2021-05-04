"""
Plot
"""

import os

from cartopy.io import img_tiles
from cartopy.mpl import ticker
from scipy.signal import find_peaks
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import seisnn.utils
import seisnn.qc


def color_palette(color=1, shade=1):
    """
    Return a color palette form a selected color and shade level.

    :param int color: (Optional.) 0=Blue, 1=Deep Orange, 2=Green, default is 1.
    :param int shade: (Optional.) 0=light, 1=regular, 2=dark, default is 1.
    :rtype: str
    :return: Hex color code.
    """
    palette = [
        ['#90CAF9', '#2196F3', '#1565C0'],  # Blue
        ['#FFAB91', '#FF5722', '#D84315'],  # Deep Orange
        ['#A5D6A7', '#4CAF50', '#2E7D32'],  # Green
    ]

    return palette[color][shade]


def get_time_array(instance):
    """
    Returns time step array from feature dict.

    :param instance: Data instance.
    :rtype: numpy.array
    :return: Time array.
    """
    time_array = np.arange(instance.metadata.npts)
    time_array = time_array * instance.metadata.delta
    return time_array


def plot_dataset(instance, title=None, save_dir=None):
    """
    Plot trace and label.

    :param instance:
    :param title:
    :param save_dir:
    """
    if title is None:
        title = f'{instance.metadata.starttime}_{instance.metadata.id[:-3]}'

    subplot = len(instance.trace.channel) + 1
    fig = plt.figure(figsize=(8, subplot * 2))

    # plot label
    ax = fig.add_subplot(subplot, 1, subplot)

    threshold = 0.5
    ax.hlines(threshold, 0, 30, lw=1, linestyles='--')
    peak_flag = []
    for i, label in enumerate([instance.label, instance.predict]):
        for j, phase in enumerate(label.phase[0:2]):
            color = color_palette(j, i)
            ax.plot(get_time_array(instance),
                    label.data[-1, :, j],
                    color=color, label=f'{phase} {label.tag}')

            peaks, _ = find_peaks(label.data[-1, :, j],
                                  distance=100,
                                  height=threshold)
            peak_flag.append(peaks)
            ax.legend()

    peak_flag = [[peak_flag[0], peak_flag[1]], [peak_flag[2], peak_flag[3]]]
    if ax.get_ylim()[1] < 1.5:
        ax.set_ylim([-0.05, 1.05])

    # plot trace
    lines_shape = [':', '-']
    for i, chan in enumerate(instance.trace.channel):
        ax = fig.add_subplot(subplot, 1, i + 1)
        ax.set_ylim([-1.05, 1.05])
        if i == 0:
            plt.title(title[0:-2])
        trace = instance.trace.data[-1, :, i]
        ax.plot(get_time_array(instance), trace, "k-", label=chan)
        for j, phase in enumerate(['label', 'predict']):
            for k, peak in enumerate(peak_flag[j]):
                color = color_palette(k, j)
                ax.vlines(peak_flag[j][k] / 100, -1.05, 1.05, color,
                          lines_shape[j])
        ax.legend(loc=1)

    if save_dir:
        seisnn.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
        plt.close()
    else:
        plt.show()


def plot_loss(log_file, save_dir=None):
    """
    Plot loss history.

    :param log_file:
    :param save_dir:
    """
    loss = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            loss.append(line)

    file_name = os.path.basename(log_file).split('.')
    loss = np.asarray(loss).astype(np.float32)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(loss[:, 0], label='train')
    ax.plot(loss[:, 1], label='validation')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    ax.legend()
    plt.title(f'{file_name[0]} loss')

    if save_dir:
        seisnn.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{file_name[0]}.png'))
        plt.close()
    else:
        plt.show()


def plot_error_distribution(time_residuals, save_dir=None):
    """
    Plot error distribution.

    :param time_residuals:
    :param save_dir:
    """
    bins = np.linspace(-0.5, 0.5, 100)
    plt.hist(time_residuals, bins=bins)
    plt.xticks(np.arange(-0.5, 0.51, step=0.1))
    plt.xlabel("Time residuals (sec)")
    plt.ylabel("Counts")
    plt.title("Error Distribution")

    if save_dir:
        seisnn.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_snr_distribution(pick_snr, save_dir=None):
    """
    Plot signal to noise ratio distribution.

    :param pick_snr:
    :param save_dir:
    """
    sns.set()
    bins = np.linspace(-1, 10, 55)
    plt.hist(pick_snr, bins=bins)
    plt.xticks(np.arange(-1, 11, step=1))
    plt.xlabel("Signal to Noise Ratio (log10)")
    plt.ylabel("Counts")
    plt.title("SNR Distribution")

    if save_dir:
        seisnn.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(true_positive, pred_count, val_count):
    """
    Plot confusion matrix.

    :param true_positive:
    :param pred_count:
    :param val_count:
    """
    matrix = pd.DataFrame([[true_positive, pred_count - true_positive],
                           [val_count - true_positive, 0]],
                          columns=['True', 'False'],
                          index=['True', 'False'])

    precision, recall, f1 = seisnn.qc.precision_recall_f1_score(true_positive,
                                                                pred_count,
                                                                val_count)

    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, cbar=False, fmt="d", cmap='Blues',
                square=True)
    bottom, top = plt.ylim()
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(
        f'Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}')
    plt.show()
    sns.set(font_scale=1)


def plot_map(geometry, events):
    """
    Plot map.

    :param geometry:
    :param events:
    """
    stamen_terrain = img_tiles.Stamen('terrain-background')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    ax.add_image(stamen_terrain, 9)

    if events:
        eq = []
        for event in events:
            eq.append([event.longitude, event.latitude])
        eq = np.array(eq).T
        ax.scatter(eq[0], eq[1],
                   label='Event',
                   transform=ccrs.PlateCarree(),
                   color='#555555',
                   edgecolors='k',
                   linewidth=0.3,
                   marker='o',
                   s=10)

    if geometry:
        geom = []
        network = geometry[0].network
        for station in geometry:
            geom.append([station.longitude, station.latitude])
        geom = np.array(geom).T
        ax.scatter(geom[0], geom[1],
                   label=network,
                   transform=ccrs.PlateCarree(),
                   color='#c72c2c',
                   edgecolors='k',
                   linewidth=0.1,
                   marker='v',
                   s=40)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    conv = ProjectionConverter(stamen_terrain.crs, ccrs.PlateCarree())
    xmin, ymin = conv.convert(xmin, ymin)
    xmax, ymax = conv.convert(xmax, ymax)

    xticks = ticker.LongitudeLocator(nbins=2)._raw_ticks(xmin, xmax)
    yticks = ticker.LatitudeLocator(nbins=3)._raw_ticks(ymin, ymax)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        ticker.LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(ticker.LatitudeFormatter())

    ax.legend()
    plt.show()


class ProjectionConverter:
    """
    Cartopy projection converter.
    """

    def __init__(self, source_proj, target_proj):
        self.x = None
        self.y = None
        self.source_proj = source_proj
        self.target_proj = target_proj

    def convert(self, x, y):
        """
        Returns converted project location.

        :param float x: X location.
        :param float y: Y location.
        :rtype: float
        :return: (x, y)
        """
        self.x = x
        self.y = y
        result = self.target_proj._project_point(self, self.source_proj)
        return result.x, result.y


if __name__ == "__main__":
    pass
