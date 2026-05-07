

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
        
if __name__ == '__main__':

    lw_ = 3
    res = 50 

    max_ = 19000
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    titles = [
        # 'SAC_on_ALE_BattleZone',
        #       'SAC_on_ALE_MontezumaRevenge',
        #       'SAC_on_ALE_NameThisGame',
        'GRPO_Sokoban',
          # 'SAC_on_Walker2d-v4',
          # 'SAC_on_Humanoid-v4',
          # 'SAC_on_HalfCheetah-v4',
              ]
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************

    for title in titles:

        fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
        # title = 'SAC_on_ALE_BattleZone'
        datadir = './data/'+title+'.csv'
        df = pd.read_csv(datadir)
        ax3.set_title(title)

        jobs = get_jobs(df, tag=" - subopt/gap_global")

        # add_plot(ax3, df, key=" - charts/replay_best_returns", label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'], lw=lw_)
        add_plot(ax3, df, key=" - subopt/policy_avg", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'], lw=lw_, key_suffix=" - train/global_step")
        add_plot(ax3, df, key=" - subopt/top_best_five", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'], lw=lw_, key_suffix=" - train/global_step")
        add_plot(ax3, df, key=" - subopt/top_best_local", label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'], lw=lw_, key_suffix=" - train/global_step")
        # add_plot(ax3, df, key=" - subopt/top_best_five", label='$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)'], lw=lw_)
        # add_plot(ax3, df, key=" - charts/deterministic_returns", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'], lw=lw_)

        render_plot(ax3, fig, title.replace('_',' '))
    
    # datadir = './data/GRPO_Sokoban.csv'
    # df = pd.read_csv(datadir)
    # title = 'Difference with GRPO on Sokobahn'
    # ax3.set_title(title)

    # df = pd.read_csv(datadir)
    # jobs = get_jobs(df, tag=" - subopt/gap_global")


    # #####################
    # ##### w/ \pi ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - subopt/gap_global", res=res, jobs=jobs, max=max_,
    #                            key_suffix=" - train/global_step")

    # label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    # ax3.lines[-1].set_linestyle(linestyle[label])


    # #####################
    # ##### w/ \pi deterministic ######
    # #####################
    
    # plot_data = get_data_frame(df, key=" - subopt/gap_local", res=res, jobs=jobs, max=max_,
    #                            key_suffix=" - train/global_step")

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

