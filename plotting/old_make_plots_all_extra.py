

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    res = 100
    lw_ = 3

    # fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    # title = 'Return on Montezumas Revenge Explore'
    # datadir = './data/PPO_MR_All.csv'
    # df = pd.read_csv(datadir)
    # ax3.set_title(title)

    # jobs = [
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851535__1__1747793950",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851536__2__1747793950",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851530__4__1747793950",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851537__3__1747793950",

    # ]

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # datadir = './data/PPO_MR-all2.csv'
    # df = pd.read_csv(datadir)

    # jobs = [
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833901__2__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833902__3__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833529__4__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833900__1__1747602512",
    # ]

    # #####################
    # ##### w/ \pi ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w RND'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w RND'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w RND'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    # ax3.set(ylabel='Return')
    # ax3.set(xlabel='Steps')
    # ax3.legend(fontsize='14')
    # fig.tight_layout(pad=0.5)
    # #plt.subplots_adjust(bottom=.25, wspace=.25)
    # plt.show()
    # fig.savefig("data/"+title+".svg")
    # fig.savefig("data/"+title+".png")
    # fig.savefig("data/"+title+".pdf")


    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    title = 'Explore Return for SpaceInvaders'
    datadir = './data/PPO_SpaceInvaders.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
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

    datadir = './data/PPO_SpaceInvaders_with_RND.csv'
    df = pd.read_csv(datadir)

    jobs = get_jobs(df)

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg) w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best) w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs)
    label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent) w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")

