

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *

if __name__ == '__main__':

    res = 50
    lw_ = 3
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/DQN_Minatar_Breakout_all.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality Gap for MinAtar Breakout'
    ax3.set_title(title)

    jobs = [
        "MinAtar/Breakout-v0__dqn__6848987__4__1747756857",
        "MinAtar/Breakout-v0__dqn__6849039__1__1747756857",
        "MinAtar/Breakout-v0__dqn__6849040__2__1747756857",
        "MinAtar/Breakout-v0__dqn__6849041__3__1747756857",
    ]

    #####################
    ##### w/ Optimal ######

    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)

    label='Best traj'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label='Global Gap'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs)

    label='Best $5\%$ from last 1000 episodes - $\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps x $10^4$')
    ax3.legend(fontsize='14')
    #plt.subplots_adjust(bottom=.25, wspace=.25)
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
    
    datadir = './data/PPO-halfCheetah.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality Gap for HalfCheetah'
    ax3.set_title(title)

    jobs = [
        "HalfCheetah-v4__ppo_continuous_action__1__1747414032",
        "HalfCheetah-v4__ppo_continuous_action__2__1747414032",
        "HalfCheetah-v4__ppo_continuous_action__3__1747414032",
        "HalfCheetah-v4__ppo_continuous_action__4__1747414032",
    ]

    #####################
    ##### w/ Optimal ######

    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label='Global Gap'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    
        #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/replay_best_returns", res=res, jobs=jobs)

    label='Best traj'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])


    #####################
    ##### w/ \pi deterministic ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/local_optimality_gap", res=res, jobs=jobs)

    label='Best $5\%$ from last 1000 episodes - $\pi$'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label], linewidth=lw_)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Return')
    ax3.set(xlabel='Steps x $10^4$')
    ax3.legend(fontsize='14')
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")
