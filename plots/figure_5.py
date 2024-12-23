import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from utils import *

def plot_figure_5a():
    """
        Visualize the double Yin-Yang variant.
    """
    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_box_aspect(1)

    ticks = [0, 0.5, 1.0, 1.5, 2.0]

    colors = ["C0", "C1", "C2"]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    log_data = np.loadtxt(f'{input_dir}/double_yinyang_dataset.txt')
    xs, ys, cs = log_data[0, :], log_data[1, :], log_data[2, :] # (x, y, class)

    ax.scatter(xs[cs == 0], ys[cs == 0], color=colors[0], edgecolor='k', alpha=0.5, label='Class 1: Yin')
    ax.scatter(xs[cs == 1], ys[cs == 1], color=colors[1], edgecolor='k', alpha=0.5, label='Class 2: Yang')
    ax.scatter(xs[cs == 2], ys[cs == 2], color=colors[2], edgecolor='k', alpha=0.5, label='Class 3: Dot')
    ax.set_xlabel('Input feature 1')
    ax.set_ylabel('Input feature 2')

    def atob(a):
        b = a+0
        return b
    
    def btoa(b):
        a = 2-b
        return a

    secax = ax.secondary_xaxis('top', functions=(atob, btoa))
    secax.set_xlabel('Input feature 3')
    secax.set_xticks(ticks)

    secax = ax.secondary_yaxis('right', functions=(atob, btoa))
    secax.set_ylabel('Input feature 4')
    secax.set_yticks(ticks)

    plt.legend()

    plt.savefig(f'{output_dir}/figure_5a.{format}', format=format, dpi=dpi)


def plot_figure_5b(mapping_scheme, encoding_scheme, sampling_iters, alphas, metric):
    """
        Mean test accuracies (or errors) for layer ensemble averaging at increasing values of redundancy parameters
        𝛼 ∈ [ , ] and 𝛽 ∈ [ , 𝛼] under different combinations of the greedy and random mapping algorithms with the simple
        and reduced mapping error (RME) encoding algorithms, where 𝛼 indicates the total number of redundant mappings
        of each conductance matrix and 𝛽 indicates how many rows (out of 𝛼) contribute to the current averaging process for
        each output.
    """

    metric_to_col_idx = {'accuracy': 2, 'mappingerror': 6}
    col_idx = metric_to_col_idx[metric]
    colors = ['#e7af2c', '#e6df2f', '#a5cc41', '#50b844', '#3dbc84', '#32bec1', '#4097d0', '#405aa7', '#624ba0', '#884496']

    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(111)

    box_data = []
    x_labels = []

    offset = 0.4 # separation b/w scatter and boxes for each data point
    LINEWIDTH = 0.75

    i = 0
    for alpha in alphas:
        for beta in range(1, alpha+1):
            # Load data
            filename = f'{input_dir}/networkmlp_yinyang_daffodil_hw_weightencoding{encoding_scheme}_xbmapping{mapping_scheme}_alpha{alpha}_beta{beta}_samplingiters{sampling_iters}_accs_errors.txt'

            data = np.loadtxt(filename)[:, col_idx] # load accuracy or mapping error as specified

            Q1 = np.percentile(data, 25)           # First quartile (25th percentile)
            median = np.median(data)               # Median (50th percentile)
            Q3 = np.percentile(data, 75)           # Third quartile (75th percentile)
            IQR = Q3 - Q1

            min_val = Q1 - 1.5 * IQR
            max_val = Q3 + 1.5 * IQR

            box_data.append({
                'whislo': min_val,   # Minimum (lower whisker)
                'q1': Q1,       # First quartile (Q1)
                'med': median,      # Median
                'q3': Q3,       # Third quartile (Q3)
                'whishi': max_val,    # Maximum (upper whisker)
                'fliers': data[data < min_val].tolist() + data[data > max_val].tolist()
            })

            x_labels.append(str((alpha, beta)))
            colors.append(colors[i])
            i += 1

    # Plot boxes
    x_positions = np.array(range(1, 2 * (len(box_data)) + 1, 2))
    bplot = ax.bxp(box_data,  
            positions=x_positions + offset,
            showfliers=True,               
            patch_artist=True,
            widths=0.5,
    )

    # Modify the boxes
    for i, (box, median) in enumerate(zip(bplot['boxes'], bplot['medians'])):
        box.set_facecolor(lighten_color(colors[i]))

        box.set_edgecolor((colors[i]))
        median.set_color(colors[i])

        # Change error bar color
        bplot['whiskers'][i*2].set_color(colors[i])  # Lower whisker
        bplot['whiskers'][i*2 + 1].set_color(colors[i])  # Upper whisker
        bplot['caps'][i*2].set_color(colors[i])  # Lower cap
        bplot['caps'][i*2 + 1].set_color(colors[i])  # Upper cap

        bplot['fliers'][i].set(marker='o', markersize=1, 
                         markerfacecolor=colors[i],  # Change fill color
                         markeredgecolor=colors[i],          # Change outline color
                         alpha=0.6)  # Set transparency

        bplot['whiskers'][i*2].set_linewidth(LINEWIDTH)  # Lower whisker line width
        bplot['whiskers'][i*2 + 1].set_linewidth(LINEWIDTH)  # Upper whisker line width
        bplot['caps'][i*2].set_linewidth(LINEWIDTH)  # Lower cap line width
        bplot['caps'][i*2 + 1].set_linewidth(LINEWIDTH)  # Upper cap line width
        median.set_linewidth(LINEWIDTH)  # Optional: Set the thickness of the median line
        box.set_linewidth(LINEWIDTH)

    # Let's do the scatter points now
    i = 0
    for alpha in alphas:
        for beta in range(1, alpha+1):
            filename = f'{input_dir}/networkmlp_yinyang_daffodil_hw_weightencoding{encoding_scheme}_xbmapping{mapping_scheme}_alpha{alpha}_beta{beta}_samplingiters{sampling_iters}_accs_errors.txt'
            d = np.loadtxt(filename)[:, col_idx]
            # Overlay scatter points jittered according to the violin plot distribution
            kde = gaussian_kde(d)
            # Evaluate KDE on a grid of points
            x_vals = np.linspace(min(d), max(d), 100)
            density = kde(x_vals)
            # Normalize the density to fit within the violin width
            max_density = density.max()
            # For each data point, plot the scatter at a position based on density
            for point in d:
                point_density = kde(point)[0]
                jitter_width = (point_density / max_density) * 0.3  # Adjust the multiplier for more jitter
                x_jittered = np.random.uniform(-jitter_width, jitter_width)
                # Plot the point at (i + x_jittered, point)
                ax.scatter(x_positions[i] - offset + x_jittered, point, alpha=0.6, s=1, color=colors[i], zorder=2)
            i+=1

    # Adding labels
    plt.xticks(rotation=45, ticks=x_positions, labels=[str(cat) for cat in x_labels])
    plt.xlabel('Redundancy, 'r'$(\alpha, \beta)$')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    if (metric == "accuracy"):
        plt.ylabel('Mean test accuracy (%)')
        plt.ylim(45, 75)
    else:
        plt.ylabel('Mapping error (%)')

    plt.xlim(0, alpha*(1+alpha))
    plt.savefig(f'{output_dir}/figure_5b_{mapping_scheme}_{encoding_scheme}_samplingiters{sampling_iters}_metric{metric}.{format}', format=format, dpi=dpi)

if __name__ == "__main__":
    
    input_dir = f'./data/figure_5'

    mapping_schemes = ['greedy', 'random']
    encoding_schemes = ['simple', 'rme']

    # Figure 5a

    plot_figure_5a()

    # Figure 5b
    # for mapping_scheme in mapping_schemes:
    #     for encoding_scheme in encoding_schemes:
    #         sampling_iters = -1 if mapping_scheme == 'greedy' else 1000
    #         # Plot accuracy
    #         plot_figure_5b(mapping_scheme=mapping_scheme, encoding_scheme=encoding_scheme, sampling_iters=sampling_iters, alphas = range(1, 5), metric='accuracy')
    #         # Plot mapping error (not part of Figure 5, can be uncommented)
    #         # plot_figure_5b(mapping_scheme=mapping_scheme, encoding_scheme=encoding_scheme, sampling_iters=sampling_iters, alphas = range(1, 5), metric='mappingerror')
    plt.show()