

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    res = 10
    lw_ = 3

    fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    title = 'Top percent for SpaceInvaders'
    datadir = './data/PPO_SpaceInvaders_top05percent.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (best replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])



    datadir = './data/PPO_SpaceInvaders_top10percent.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w top10'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w top10'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w top10'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/PPO_SpaceInvaders_top20percent.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w top20'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w top20'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w top20'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    

    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ## Increase fontsize of ticks
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.set(ylabel='Return')
    ## increase fontsize of labels
    ax3.set_ylabel('Return', fontsize=18)
    ax3.set_title(title, fontsize=20)
    ax3.set_xlabel('Steps', fontsize=18)
    ax3.legend(fontsize='18')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")


    fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    title = 'Top percent for Asterix'
    datadir = './data/PPO_Asterix_top05percent.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*} }(s_0)$ (best replay)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    

    datadir = './data/PPO_Asterix_top10percent.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w top10'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w top10'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w top10'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/PPO_Asterix_top20percent.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w top20'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w top20'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w top20'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    

    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ## Increase fontsize of ticks
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.set(ylabel='Return')
    ## increase fontsize of labels
    ax3.set_ylabel('Return', fontsize=18)
    ax3.set_title(title, fontsize=20)
    ax3.set_xlabel('Steps', fontsize=18)
    ax3.legend(fontsize='18')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")

    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
    
    title = 'Top percent for LunarLander'
    datadir = './data/PPO_LunarLander_top05percent.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])



    datadir = './data/PPO_LunarLander_top10percent.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w top10'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w top10'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w top10'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/PPO_LunarLander_top20percent.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w top20'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w top20'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w top20'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    

    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ## Increase fontsize of ticks
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.set(ylabel='Return')
    ## increase fontsize of labels
    ax3.set_ylabel('Return', fontsize=18)
    ax3.set_title(title, fontsize=20)
    ax3.set_xlabel('Steps', fontsize=18)
    ax3.legend(fontsize='18')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")

