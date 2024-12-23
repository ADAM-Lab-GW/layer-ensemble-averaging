import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from utils import *

def plot_figure_4b(alphas=range(1, 7), metric='accuracy'):
    """
        Network test accuracies and mapping errors averaged across network layers at
        increasing levels of device stuck-at faults and the redundancy parameter 𝛼 for each fault tolerance scheme. 
        Bar heights correspond to averages and error bars correspond to standard deviations across 10 independent cycles of the entire simulation process. 
    """

    encoding_schemes = ['LEA1', 'MAO', 'CM',]
    encoding_schemes_nice_names = ['LEA - Simple', 'MAO', 'CM']
    colors = ['#c21a09',  '#9897A9', '#c5c6d0',]

    if (metric == "mappingerror"):
        for i in range(len(colors)):
            colors[i] = lighten_color(colors[i], amount=1.1)

    for alpha in alphas:
        fig = plt.figure(figsize=figsize, layout='compressed')
        ax = fig.add_subplot(111)
        ax.set_box_aspect(1)

        ax.grid(zorder=0, axis="x", color = "lightgrey", linewidth = "0.5")
        plot_legend = alpha == 1

        # at a fixed alpha, gather average data points for each encoding_scheme
        algo_means = {}
        algo_stds = {}
        for i, encoding_scheme in enumerate(encoding_schemes):
            # load data
            data = np.loadtxt(f"{input_dir}/{metric}_comparison_alpha{alpha}_{encoding_scheme}.txt")
            stuck_percentages, data_mean, data_std = data[0, :], data[1, :], data[2, :]
            algo_means[encoding_schemes_nice_names[i]] = data_mean
            algo_stds[encoding_schemes_nice_names[i]] = data_std

        x = np.arange(len(stuck_percentages))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        for attribute, measurement in algo_means.items():
            offset = width * multiplier
            rects = ax.barh(x + offset, measurement, width, xerr=algo_stds[attribute], label=attribute, color=colors[multiplier], error_kw={'capsize':1, 
                                                                                                                                            'elinewidth':0.5}, zorder=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if (metric == "mappingerror"): ax.set_xlabel('Mapping Error (%)')
        else: ax.set_xlabel('Test Accuracy (%)')

        ax.set_yticks(x + width, [s * 100 for s in stuck_percentages])

        if (metric == "accuracy"): 
            # Plot accuracy bars sideways
            ax.set_xlim(100, 0)
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("right")

            ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both')
            
            ax.xaxis.set_minor_locator(MultipleLocator(2))
            ax.xaxis.set_major_locator(MultipleLocator(10))

            plt.axvline(x=sw_baseline[0], color='gray', linestyle="dotted", linewidth=0.8, label='Software Baseline', zorder=-1)

            if (plot_legend):
                ax.legend(loc='upper center')

        elif (metric == "mappingerror"): 
            ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both')
            ax.set_xlim(0, 500) # mapping error
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.xaxis.set_major_locator(MultipleLocator(50))

            if (plot_legend):
                ax.legend()

        ax.set_ylabel('Stuck percentage (%)')
        plt.savefig(f'{output_dir}/figure_4b_{metric}_alpha{alpha}.{format}', format=format, dpi=dpi)

if __name__ == "__main__":
    
    input_dir = f'./data/figure_4'

    sw_baseline = (94.49, 0.12)

    plot_figure_4b(metric='accuracy')
    plot_figure_4b(metric='mappingerror')