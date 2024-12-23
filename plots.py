import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from pathlib import Path

def plot_figure_3b(vread=0.3, std_scaling=3):
    """
        Device retention data showcasing effective 2-bit tuning of a representative device. 
        The error bars indicate three standard deviations.
    """
    alpha = 0.2
    colors = [(0, 0, 1, alpha), (0.7, 0.7, 0.7, alpha), (0.7, 0.7, 0.7, alpha), (1, 0, 0, alpha)]  # Red to Blue
    labels = ['133 μS ($\mathit{G_{OFF}}$)', '167 μS', '200 μS', '233 μS ($\mathit{G_{ON}}$)']

    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(1, 1, 1)

    for idx in range(len(labels)):
        # Load data and plot
        data = np.loadtxt(f'./data/figure_3/retention_grouped_state{idx}.txt')
        time, conductance, errs = data[:, 0], data[:, 1], data[:, 2] * std_scaling
        
        plt.errorbar(time, conductance, yerr=errs, fmt='o', alpha=alpha, label=labels[idx], c=colors[idx], mec='k', ms=5)


    ax.set_ylabel('Conductance, $\mathit{G}$ (μS)')
    ax.set_xlabel('Time, $\mathit{t}$ (minutes)')
    plt.legend(title=' $\mathit{G_{req}}$')
    plt.savefig(f'{output_dir}/figure_3b.{format}', format=format, dpi=dpi)

def plot_figure_3c(cycles=20, scale=1e6):
    """
        Current vs. voltage (I-V) sweep curves over multiple cycles of a single representative device.
    """
    # Load data
    allPosData = []
    allNegData = []

    for i in range(1, cycles):
        # load +ve and -ve IV sweeps at a given cycle for plotting
        data = np.loadtxt(f'./data/figure_3/iv_positive_cycle{i}.txt')
        allPosData.append(data)

        data = np.loadtxt(f'./data/figure_3/iv_negative_cycle{i}.txt')
        allNegData.append(data)

    # Plot positive side of the data
    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_box_aspect(2)
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(100))

    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both')

    set_color = 'red'
    reset_color = 'blue'
    linewidth = 0.5
    alphas = np.linspace(0.1, 0.75, num=cycles, endpoint=True)

    for cycle in range(len(allPosData)):
        plt.plot(allPosData[cycle][:, 0], allPosData[cycle][:, 1] * scale, c=reset_color, alpha=alphas[cycle], linewidth=linewidth)

    plt.ylim(-550, 450)

    plt.xlabel('Col. Voltage, $\mathit{V_{col}}$ (V)')
    plt.ylabel('Current, $\mathit{I}$ (μA)')
    plt.axhline(0, color='black', linewidth=linewidth/2, alpha=0.25)
    plt.axvline(0, color='black', linewidth=linewidth/2, alpha=0.25)

    plt.savefig(f'{output_dir}/figure_3c_1.{format}', format=format, dpi=dpi)

    # Plot negative side of the data
    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_box_aspect(1.5)

    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(100))

    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both')
    for cycle in range(len(allNegData)):
        plt.plot(allNegData[cycle][:, 0],  allNegData[cycle][:, 1] * scale, c=set_color, alpha=alphas[cycle], linewidth=linewidth)

    plt.ylim(-550, 450)

    plt.xlabel('Gate Voltage, $\mathit{V_{gate}}$ (V)')
    plt.ylabel('Current, $\mathit{I}$ (μA)')

    plt.axhline(0, color='black', linewidth=linewidth/2, alpha=0.25)
    plt.axvline(0, color='black', linewidth=linewidth/2, alpha=0.25)

    plt.savefig(f'{output_dir}/figure_3c_2.{format}', format=format, dpi=dpi)

def plot_figure_3d(kernel=17, scale=1e-6, stuck_on_threshold=300):
    """
        Conductance map of devices in a kernel randomly programmed to one of four conductance states from Figure 3b.
    """
    # Load data, convert units to μS
    data = np.loadtxt(f"./data/figure_3/conductance_map_multibit_k{kernel}.txt")
    data = data * scale

    # Plot conductance map
    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('Column #')
    plt.ylabel('Row #')

    xedges = list(range(data.shape[0]))
    yedges = list(range(data.shape[1]))
    surf = plt.pcolormesh(xedges, yedges, data.T, alpha=1, antialiased=True, linewidth=0.0, zorder=-1, vmin=0, vmax=stuck_on_threshold)
    surf.set_edgecolor('face')
    plt.axis('image')

    cbar = plt.colorbar()
    cbar.set_label('Conductance, $\mathit{G}$ (μS)')
    plt.savefig(f'{output_dir}/figure_3d.{format}', format=format, dpi=dpi)

    # Quick way to estimate average conductance of stuck ON devices for a particular kernel
    stuck_mask = data > stuck_on_threshold
    print(f'# of stuck ON devices for kernel {kernel}:', np.sum(data > stuck_on_threshold))
    print(f'Average stuck ON conductance for kernel {kernel} (μS):', np.average(data[stuck_mask]))

def plot_figure_3e(kernel=17, scale=1e-6, bins=40):
    """
        Conductance distributions of devices in a kernel randomly programmed to one of four conductance states from Figure 3b.
    """
    # Load data
    states = []
    for state in range(4):
        data = np.loadtxt(f'./data/figure_3/conductance_hist_multibit_k{kernel}_state{state}.txt')#conductance_hist_multibit_k17_state0
        states.append(data)
    
    # Plot
    alpha = 0.5
    colors = [(0, 0, 1, alpha), (0.7, 0.7, 0.7, alpha), (0.7, 0.7, 0.7, alpha), (1, 0, 0, alpha)]  # Red to Blue
    labels = ['133 μS ($\mathit{G_{OFF}}$)', '167 μS', '200 μS', '233 μS ($\mathit{G_{ON}}$)']

    plt.figure(figsize=figsize, layout='compressed')
    for idx, state in enumerate(states):
        plt.hist(state, bins=bins, color=colors[idx], label=labels[idx], edgecolor='black')
    
    plt.xlabel('Conductance, $\mathit{G}$ (μS)')
    plt.ylabel('Frequency (#)')
    plt.xlim(66.67, 300)
    plt.legend(title=' $\mathit{G_{req}}$')
    
    plt.savefig(f'{output_dir}/figure_3e.{format}', format=format, dpi=dpi)

def plot_figure_3f(kernel=17, mode='backward'):
    """
        Comparison of theoretical accumulated currents and experimental accumulated currents on the kernel for 100 randomly generated input voltage
        vectors repeated for a total of 20 iterations.
    """

    # Load data
    itheory = np.loadtxt(f'./data/figure_3/mac_itheory_k{kernel}_mode{mode}.txt')
    iexp = np.loadtxt(f'./data/figure_3/mac_iexp_k{kernel}_mode{mode}.txt')

    # Plot
    markersize = 5
    alpha = 0.25
    color = 'red'
    linewidth = 0.25

    fig = plt.figure(figsize=figsize, layout='compressed')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_box_aspect(1)

    norm = np.max(abs(itheory)) # normalize by the max theoretical accumulated current

    plt.axhline(0, color='black', linewidth=linewidth, alpha=alpha)
    plt.axvline(0, color='black', linewidth=linewidth, alpha=alpha)

    ax.scatter(itheory / norm, iexp  / norm, alpha=alpha, s=markersize, c=color) # correction corresponding to the average blanket decrease in experimentally observed currents
    
    ax.set_xlabel('Theoretical Current (Norm.)')
    ax.set_ylabel('Experimental Current (Norm.)')
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.savefig(f'{output_dir}/figure_3f.{format}', format=format, dpi=dpi)

if __name__ == "__main__":

    format = 'svg'
    dpi = 1200
    figsize = (4, 3)
    output_dir = './generated_plots'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Figure 3
    plot_figure_3b()
    plot_figure_3c()
    plot_figure_3d()
    plot_figure_3e()
    plot_figure_3f()

    plt.show()