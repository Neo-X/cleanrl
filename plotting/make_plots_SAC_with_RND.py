

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
       
if __name__ == '__main__':

    res = 10
    lw_ = 3
    titles = [
        # 'SAC_on_ALE_BattleZone',
        #       'SAC_on_ALE_MontezumaRevenge',
        #       'SAC_on_ALE_NameThisGame',
        #       'SAC_on_MinAtar_Asterix',
        #       'SAC_on_MinAtar_Breakout',
        #       'SAC_on_MinAtar_Freeway',
        #       'SAC_on_MinAtar_Seaquest',
              ['SAC_on_MinAtar_SpaceInvaders', 'SAC_with_RND_on_MinAtar_SpaceInvaders' ]
              # 'SAC_on_LunarLander',
            # 'SAC_on_Walker2d-v4',
            # 'SAC_on_Humanoid-v4',
            # 'SAC_on_HalfCheetah-v4',
              ]
    for titles in titles:

        fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
        # title = 'SAC_on_ALE_BattleZone'
        datadir = './data/'+titles[1]+'.csv'
        df = pd.read_csv(datadir)
        ax3.set_title(titles[1])

        jobs = get_jobs(df)
        add_plot(ax3, df, key=" - charts/global_optimality_gap", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w RND', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best) w RND'], lw=lw_)
        add_plot(ax3, df, key=" - charts/local_optimality_gap", label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w RND', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent) w RND'], lw=lw_)
        datadir = './data/'+titles[0]+'.csv'
        df = pd.read_csv(datadir)
        jobs = get_jobs(df)
        add_plot(ax3, df, key=" - charts/global_optimality_gap", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (best)'], lw=lw_)
        add_plot(ax3, df, key=" - charts/local_optimality_gap", label='$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ (recent)'], lw=lw_)
        # add_plot(ax3, df, key=" - charts/replay_top_k_returns_stochastic", label='$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)'], lw=lw_)
        # add_plot(ax3, df, key=" - charts/deterministic_returns", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'], lw=lw_)

        render_plot(ax3, fig, titles[1].replace('_',' '))

    
    