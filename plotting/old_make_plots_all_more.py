

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    res = 100
    lw_ = 3
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    datadir = './data/DQN_BattleZone.csv'
    df = pd.read_csv(datadir)
    title = 'Performance on BattleZone'
    ax3.set_title(title)

    jobs = get_jobs(df)

    #####################
    ##### w/ Optimal ######

    
    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)

    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ $\pi$ deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/deterministic_returns", res=res, jobs=jobs)

    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ Best $5\%$ - $\pi$ ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    #####################
    ##### w/ Best $5\%$ from last 1000 episodes - $\pi$ ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    # plt.subplots_adjust(bottom=.25, wspace=.25)
    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")


    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    datadir = './data/DQN_NameThisGame.csv'
    df = pd.read_csv(datadir)
    title = 'Performance on NameThisGame'
    ax3.set_title(title)

    jobs = get_jobs(df)

    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    plot_data = get_data_frame(df, key=" - charts/deterministic_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
        
    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")


    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/DQN_Asterix.csv'
    df = pd.read_csv(datadir)
    title = 'Performance for Asterix'
    ax3.set_title(title)

    jobs = get_jobs(df)

    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    plot_data = get_data_frame(df, key=" - charts/deterministic_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
        
    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")


   

