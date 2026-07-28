

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

from make_plots import *

if __name__ == '__main__':

    res = 20
    lw_ = 3
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/PPO_MR-all.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality Gap for Montezumas Revenge'
    ax3.set_title(title)

    jobs = [
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__0__1747414034",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__6__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__5__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__4__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__3__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__1__1747414032",
        "MontezumaRevengeNoFrameskip-v4__ppo_atari__0__0__1747414032"
    ]

    #####################
    ##### w/ Optimal ######

    # steps_ = deNan(df["MontezumaRevengeNoFrameskip-v4__ppo_atari__0__1__1747414032 - global_step"].to_numpy())

    # steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
    # plot_data = pd.DataFrame([(step_, 50) for step_ in steps_])

    # label='Expert'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label])
    # ax3.lines[-1].set_linestyle(linestyle[label])

    
    #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label='Best $5\%$ - $\pi$'
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
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    fig.tight_layout(pad=0.5)
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

    # steps_ = deNan(df["MontezumaRevengeNoFrameskip-v4__ppo_atari__0__1__1747414032 - global_step"].to_numpy())

    # steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
    # plot_data = pd.DataFrame([(step_, 50) for step_ in steps_])

    # label='Expert'
    # plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    # sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label, c=colors[label])
    # ax3.lines[-1].set_linestyle(linestyle[label])

    
        #####################
    ##### w/ \pi ######
    #####################
    
    plot_data = get_data_frame(df, key=" - charts/global_optimality_gap", res=res, jobs=jobs)

    label='Best $5\%$ - $\pi$'
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
    ax3.set(xlabel='Steps')
    ax3.legend(fontsize='14')
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")
