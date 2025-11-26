

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *

def add_plot(ax, df, key, label, res, jobs, color, lw):
    plot_data = get_data_frame(df, key=key, res=res, jobs=jobs)
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax, label=label, c=color, linewidth=lw)
    ax.lines[-1].set_linestyle(linestyle[label])

def render_plot(ax, title):

    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ## Increase fontsize of ticks
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.set(ylabel='Return')
    ## increase fontsize of labels
    ax3.set_ylabel('Return', fontsize=18)
    ax3.set_title(title, fontsize=20)
    ax3.set_xlabel('Steps', fontsize=18)
    ## make the legend more see through
    ax3.get_legend().get_frame().set_alpha(0.2)
    ax3.legend(fontsize='16')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    title2 = title.replace(" ","_").replace(".","").replace("/","_")
    fig.savefig("data/"+title2+".svg")
    fig.savefig("data/"+title2+".png")
    fig.savefig("data/"+title2+".pdf")
        
if __name__ == '__main__':

    res = 10
    lw_ = 3
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
    title = 'SAC on Humanoid'
    datadir = './data/SAC_Humanoid-v4.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    add_plot(ax3, df, key=" - charts/replay_best_returns", label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/episodic_return", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/avg_top_returns_global", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/avg_top_returns_local", label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/replay_top_k_returns_stochastic", label='$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/deterministic_returns", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'], lw=lw_)

    render_plot(ax3, title)

    ################################################################################
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
    title = 'SAC on Walker2d'
    datadir = './data/SAC_Walker2d-v4.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    add_plot(ax3, df, key=" - charts/replay_best_returns", label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/episodic_return", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/avg_top_returns_global", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/avg_top_returns_local", label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/replay_top_k_returns_stochastic", label='$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/deterministic_returns", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'], lw=lw_)

    render_plot(ax3, title)

    ################################################################################
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
    title = 'SAC on HalfCheetah'
    datadir = './data/SAC_HalfCheetah-v4.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    add_plot(ax3, df, key=" - charts/replay_best_returns", label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/episodic_return", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/avg_top_returns_global", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/avg_top_returns_local", label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/replay_top_k_returns_stochastic", label='$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)'], lw=lw_)
    add_plot(ax3, df, key=" - charts/deterministic_returns", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'], lw=lw_)

    render_plot(ax3, title)
    