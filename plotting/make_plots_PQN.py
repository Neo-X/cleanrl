

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *
       
if __name__ == '__main__':

    res = 10
    lw_ = 3
    titles = [
            # 'PQN_on_ALE_BattleZone',
            #   'PQN_on_ALE_MontezumaRevenge',
            #   'PQN_on_ALE_NameThisGame',
            #   'PQN_on_MinAtar_Asterix',
            #   'PQN_on_MinAtar_Breakout',
            #   'PQN_on_MinAtar_Freeway',
            #   'PQN_on_MinAtar_Seaquest',
              'PQN_on_MinAtar_SpaceInvaders',
              'PQN_on_LunarLander'
              ]
    for title in titles:

        fig, (ax3) = plt.subplots(1, 1, figsize=(8,4.5))
        # title = 'PQN_on_ALE_BattleZone'
        datadir = './data/'+title+'.csv'
        df = pd.read_csv(datadir)
        ax3.set_title(title)

        jobs = get_jobs(df)

        add_plot(ax3, df, key=" - charts/replay_best_returns", label='$V^{ \hat{\pi}^{*} }(s_0)$ (replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (replay)'], lw=lw_)
        add_plot(ax3, df, key=" - charts/episodic_return", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (avg)'], lw=lw_)
        add_plot(ax3, df, key=" - charts/avg_top_returns_global", label='$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$ (best)'], lw=lw_)
        add_plot(ax3, df, key=" - charts/avg_top_returns_local", label='$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*}_{D} }(s_0)$ (recent)'], lw=lw_)
        add_plot(ax3, df, key=" - charts/replay_top_k_returns_stochastic", label='$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{*} }(s_0)$ (top replay)'], lw=lw_)
        add_plot(ax3, df, key=" - charts/deterministic_returns", label='$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)', res=res, jobs=jobs, color=colors['$V^{ \hat{\pi}^{\\theta} }(s_0)$ (deterministic)'], lw=lw_)

        render_plot(ax3, fig, title.replace('_',' '))

    
    