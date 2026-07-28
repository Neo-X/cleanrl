

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
    
    # datadir = './data/PPO_MR-all2.csv'
    # df = pd.read_csv(datadir)
    # title = 'Return on Montezumas Revenge'
    # ax3.set_title(title)

    # jobs = [
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833901__2__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833902__3__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833529__4__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833900__1__1747602512",
    # ]

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/deterministic_returns", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
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
        
    # # plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    # # label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    # # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    
    # ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    # ax3.set(ylabel='Return')
    # ax3.set(xlabel='Steps')
    # ax3.legend(fontsize='14')
    # # plt.subplots_adjust(bottom=.25, wspace=.25)
    # fig.tight_layout(pad=0.5)
    # plt.show()
    # fig.savefig("data/"+title+".svg")
    # fig.savefig("data/"+title+".png")
    # fig.savefig("data/"+title+".pdf")


    # fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    # datadir = './data/PPO-halfCheetah.csv'
    # df = pd.read_csv(datadir)
    # title = 'Return on HalfCheetah'
    # ax3.set_title(title)

    # jobs = [
    #     "HalfCheetah-v4__ppo_continuous_action__1__1747414032",
    #     "HalfCheetah-v4__ppo_continuous_action__2__1747414032",
    #     "HalfCheetah-v4__ppo_continuous_action__3__1747414032",
    #     "HalfCheetah-v4__ppo_continuous_action__4__1747414032",
    # ]

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/deterministic returns", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
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
        
    # # plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    # # label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    # # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    
    # ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    # ax3.set(ylabel='Return')
    # ax3.set(xlabel='Steps')
    # ax3.legend(fontsize='14')
    # fig.tight_layout(pad=0.5)
    # plt.show()
    # fig.savefig("data/"+title+".svg")
    # fig.savefig("data/"+title+".png")
    # fig.savefig("data/"+title+".pdf")


    # res = 1000
    # fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    # datadir = './data/DQN_Minatar_Breakout_all.csv'
    # df = pd.read_csv(datadir)
    # title = 'Return on MinAtar Breakout'
    # ax3.set_title(title)

    # jobs = [
    #     "MinAtar/Breakout-v0__dqn__6848987__4__1747756857",
    #     "MinAtar/Breakout-v0__dqn__6849039__1__1747756857",
    #     "MinAtar/Breakout-v0__dqn__6849040__2__1747756857",
    #     "MinAtar/Breakout-v0__dqn__6849041__3__1747756857",
    # ]

    # plot_data = get_data_frame(df, key=" - charts/episodic_return", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/deterministic_returns", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
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
        
    # plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    # ax3.set(ylabel='Return')
    # ax3.set(xlabel='Steps')
    # ax3.legend(fontsize='14',
    #     #         bbox_to_anchor=(0.5, -0.2), ncol=1,
    #     #    borderaxespad=0.0
    #        )
    # fig.tight_layout(pad=0.5)
    # #plt.subplots_adjust(bottom=.25, wspace=.25)
    # plt.show()
    # fig.savefig("data/"+title+".svg")
    # fig.savefig("data/"+title+".png")
    # fig.savefig("data/"+title+".pdf")


    res = 1000
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/DQN_MinAtar_SpaceInvaders.csv'
    df = pd.read_csv(datadir)
    title = 'Return on MinAtar Space Invaders'
    ax3.set_title(title)

    jobs = [
        "MinAtar/SpaceInvaders-v0__dqn__6849010__2__1747756827",
        "MinAtar/SpaceInvaders-v0__dqn__6849009__1__1747756827",
        "MinAtar/SpaceInvaders-v0__dqn__6848977__4__1747756828",
        "MinAtar/SpaceInvaders-v0__dqn__6849011__3__1747756827",
    ]

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
    ax3.legend(fontsize='14'
        #        , bbox_to_anchor=(0.5, -0.2), ncol=1,
        #    borderaxespad=0.0
           )
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")




    res = 10
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/CraftTax_PPO_EntCoff_0.01.csv'
    df = pd.read_csv(datadir)
    jobs = get_jobs(df)
    title = 'Return on Craftax'
    ax3.set_title(title)


    plot_data = get_data_frame(df, key=" - episode_return", res=res, jobs=jobs, global_key="Step", scale_=30000)
    label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/deterministic_returns", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_global", res=res, jobs=jobs, global_key="Step", scale_=30000)
    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    plot_data = get_data_frame(df, key=" - charts/avg_top_returns_local", res=res, jobs=jobs, global_key="Step", scale_=30000)
    label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
        
    # plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)
    # label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14', 
            # bbox_to_anchor=(0.5, -0.2), ncol=1,
        #    borderaxespad=0.0
           )
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")


    

