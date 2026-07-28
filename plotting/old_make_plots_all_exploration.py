

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    lw_ = 3
    res = 10
    # fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # # legend_details = {fontsize='18'}
    #         # bbox_to_anchor=(0.5, -0.2), ncol=1,
    #     #    borderaxespad=0.0
            
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    # datadir = './data/CraftTax_PPO_EntCoff_0.01.csv'
    # df = pd.read_csv(datadir)
    # jobs = get_jobs(df)
    # title = 'PPO Difference on Craftax'
    # ax3.set_title(title)

    # ## Normal PPO
    # plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, global_key="Step", scale_=30000)

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, global_key="Step", scale_=30000)

    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # ## PPO with RND
    # datadir = './data/CraftTax_PPO_EntCoff_0.01_wRND.csv'
    # df = pd.read_csv(datadir)
    # jobs = get_jobs(df)
    
    # plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, global_key="Step", scale_=30000)

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w RND'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, global_key="Step", scale_=30000)

    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w RND'
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



    # res = 50
    # max_ = 150000
    # fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    # #*******************************************************************************
    # #####################
    # ##### Global Optimality #####
    # #####################
    # #*******************************************************************************
    
    # datadir = './data/PPO_MR-all2.csv'
    # df = pd.read_csv(datadir)
    # title = 'Difference with PPO on Montezumas Revenge'
    # ax3.set_title(title)

    # jobs = [
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833901__2__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833902__3__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833529__4__1747602512",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833900__1__1747602512",
    # ]

    # #####################
    # ##### w/ \pi ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w RND'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    # #####################
    # ##### w/ \pi deterministic ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w RND'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])

    # # datadir = './data/PPO_MR_all2_without_RND.csv'
    # datadir = './data/PPO_MR_All.csv'
    # df = pd.read_csv(datadir)
    # ax3.set_title(title)

    # jobs = [
    #     # "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833417__2__1747595815",
    #     # "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833418__3__1747595815",
    #     # "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833416__1__1747595815",
    #     # "MontezumaRevengeNoFrameskip-v4__ppo_atari__6833409__4__1747595815",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851535__1__1747793950",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851536__2__1747793950",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851530__4__1747793950",
    #     "MontezumaRevengeNoFrameskip-v4__ppo_atari__6851537__3__1747793950",

    # ]

    # #####################
    # ##### w/ \pi ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    # #####################
    # ##### w/ \pi deterministic ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs)

    # label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    # ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    # ax3.set(ylabel='Return')
    # ax3.set(xlabel='Steps')
    # ax3.legend(fontsize='14', 
    #         # bbox_to_anchor=(0.5, -0.2), ncol=1,
    #     #    borderaxespad=0.0
    #        )
    # fig.tight_layout(pad=0.5)
    # #plt.subplots_adjust(bottom=.25, wspace=.25)
    # plt.show()
    # fig.savefig("data/"+title+".svg")
    # fig.savefig("data/"+title+".png")
    # fig.savefig("data/"+title+".pdf")



    max_ = 19000
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/PPO_SpaceInvaders_with_RND.csv'
    df = pd.read_csv(datadir)
    title = 'Difference with PPO on SpaceInvaders'
    ax3.set_title(title)

    jobs = [
        "SpaceInvadersNoFrameskip-v4__ppo_atari__6833537__4__1747603416",
        "SpaceInvadersNoFrameskip-v4__ppo_atari__6834014__3__1747603275",
        "SpaceInvadersNoFrameskip-v4__ppo_atari__6834007__2__1747603238",
        "SpaceInvadersNoFrameskip-v4__ppo_atari__6834006__1__1747603238",
        
    ]

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs, max=max_)

    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs, max=max_)

    label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w RND'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])

    datadir = './data/PPO_SpaceInvaders.csv'
    df = pd.read_csv(datadir)
    ax3.set_title(title)

    jobs = get_jobs(df)

    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs)

    label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
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

