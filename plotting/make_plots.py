

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

colors = {'$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$': '#bf5b17',
          '$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$' : '#386cb0',
          '$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w RND': '#B6992D',
          '$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w RND': '#7fc97f',
          '$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w ResNet': "#beaed4",
          '$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w ResNet': "#ffff99",
          'Expert': '#D7B5A6',
          '$V^{ \hat{\pi}^{*} }(s_0)$' : "#C36FC3",
          '$V^{ \hat{\pi}^{\\theta} }(s_0)$' : '#666666',
          '$V^{ \hat{\pi}^{\\theta} }(s_0)$ deterministic': '#f0027f',
          '$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$': '#A0CBE8',
          '$V^{ \hat{\pi}^{*} }(s_0)$': '#E15759',
         }
linestyle = {'$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$': '-',
          '$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$': '-',
          '$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w RND': '--',
          '$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w RND': '--',
          '$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w ResNet': "--",
          '$V^{ \hat{\pi}^{*}_{D} }(s_0) - V^{ \hat{\pi}^{\\theta} }(s_0)$ w ResNet': "--",
          'Expert': '-',
          '$V^{ \hat{\pi}^{*} }(s_0)$': '-',
          '$V^{ \hat{\pi}^{*}_{ D_{\infty} } }(s_0)$' : '-',
          'Reward + ICM' : '-',
          '$V^{ \hat{\pi}^{\\theta} }(s_0)$': '-',
          '$V^{ \hat{\pi}^{\\theta} }(s_0)$ deterministic': '--',
          '$V^{ \hat{\pi}^{*} }(s_0)$': '--',
         }
def plotsns_smoothed(ax, s, df, label, title=None, ylabel=None, res=1):
    data = list(df[s])
    s = s.split('/')[-1]
    data = pd.DataFrame([(i//res*res, data[i]) for i in range(len(data))])
    data = data.rename(columns={1: s, 0: 'Episodes'})
    ax = sns.lineplot(x='Episodes', y=s, data=data, label=label, ax=ax, legend=False,  c=colors[label])

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(s)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

def plotsns(ax, s, df, label, title=None, ylabel=None, res=1):
    data = list(df[s])
    data = (np.cumsum(data)[res:]-np.cumsum(data)[:-res]) / res
    s = s.split('/')[-1]
    data = pd.DataFrame([(i, data[i]) for i in range(len(data))])
    data = data.rename(columns={1: s, 0: 'Episodes'})
    ax = sns.lineplot(x='Episodes', y=s, data=data, label=label, ax=ax, legend=False, c=colors[label])
    #print(ax.lines)
    ax.lines[-1].set_linestyle(linestyle[label])
    #print(\label\, label,linestyle[label] )
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(s)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    ax.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    #ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)

def save(fname):
    plt.show()
    '''
    plt.savefig('{}.png'.format(fname))
    plt.clf()
    '''

def deNan(data_):
    if np.isnan(data_[0]):
        ## Search foreward for a value and replace the first values
        i = 0
        while np.isnan(data_[i]):
            i += 1
        data_[:i] = data_[i]

    ## Repalce all NaN values with the last value
    for i in range(len(data_)): 
        if np.isnan(data_[i]):
            data_[i] = data_[i-1]
    return data_

def get_data_frame(df, key, res=10, jobs=None, max=10000000000):
    

    plot_data = []
    for i in range(len(jobs)): 
        key__ =   jobs[i]+" - global_step"
        len_ = min(len(df[key__]), max)
        steps_ = range(len_)
        steps_t = deNan(df[key__].to_numpy())[:len_]
        scale_ = steps_t[-1] / steps_[-1]
        steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
        data_ = df[jobs[i]+key][:max].to_numpy()
        data_ = deNan(data_)
        print(jobs[i]+key, data_)
        # data_ = (np.cumsum(data_)[res:] - np.cumsum(data_)[:-res])/res
        
        ## Average over the last 5 values with the resulting array being one 5th shorter
        data_ = np.array([np.mean(data_[i:i+res]) for i in range(0, len(data_)-res+1, res)])
        # stds_ = np.array([np.std(data_[i:i+res]) for i in range(0, len(data_)-res+1, res)])


        # plot_data.extend([(step_, val, std_) for step_, val, std_ in zip(steps_, data_, stds_)])
        plot_data.extend([(step_, val) for step_, val in zip(steps_, data_)])
    
    ## Scale the steps based on the true data.
    plot_data = [(int(step_*scale_), val) for step_, val in plot_data]
    
    plot_data = pd.DataFrame(plot_data)
    
    return plot_data

## This function will process the data from a csv file, checking the colum keys and return the strings for those keys
def get_jobs(df):
    keys = []
    for i in range(len(df.columns)):
        key = df.columns[i]
        if ' - charts/global_optimality_gap' in key and "__MIN" not in key and "__MAX" not in key and (len(df[key]) > 10):
            #remove the end of the key
            key_ = key.split(' - charts/global_optimality_gap')[0]
            keys.append(key_)
    return keys
        
if __name__ == '__main__':

    res = 50
    fig, (ax3) = plt.subplots(1, 1, figsize=(8,5))
    #*******************************************************************************
    #####################
    ##### Global Optimality #####
    #####################
    #*******************************************************************************
    
    datadir = './data/SpaceInvaders_MinAtar_global_optimality_gap.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality for SpaceInvaders'
    ax3.set_title(title)

    #####################
    ##### w/ Optimal ######

    steps_ = df["Step"].to_numpy()
    steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
    plot_data = pd.DataFrame([(step_, 500) for step_ in steps_])

    label='Oracle'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    ax3.lines[-1].set_linestyle(linestyle[label])

    
    #####################
    ##### w/ DQN ######
    #####################
    
    keys_ = ["MinAtar/SpaceInvaders-v0__dqn__1__1745789971 - charts/global_optimality_gap",
             "MinAtar/SpaceInvaders-v0__dqn__2__1745789970 - charts/global_optimality_gap",
             "MinAtar/SpaceInvaders-v0__dqn__3__1745789971 - charts/global_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='DQN'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    # #####################
    # ##### w/ PPO ######
    # #####################
    
    keys_ = ["MinAtar/SpaceInvaders-v0__ppo__3__1745790012 - charts/global_optimality_gap",
             "MinAtar/SpaceInvaders-v0__ppo__2__1745790012 - charts/global_optimality_gap",
             "MinAtar/SpaceInvaders-v0__ppo__1__1745790012 - charts/global_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='PPO'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label} )
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Global Optimality Gap')
    ax3.set(xlabel='Steps')
    ax3.legend()
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
    
    datadir = './data/MinAtar_Asterix_global_optimality_gap.csv'
    df = pd.read_csv(datadir)
    title = 'Global Optimality for Asterix'
    ax3.set_title(title)

    #######################
    ##### w/ Optimal ######

    steps_ = df["Step"].to_numpy()
    steps_ = np.array([np.mean(steps_[i:i+res]) for i in range(0, len(steps_)-res+1, res)])
    plot_data = pd.DataFrame([(step_, 500) for step_ in steps_])

    label='Oracle'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    #####################
    ##### w/ DQN ######
    #####################
    
    keys_ = ["MinAtar/Asterix-v0__dqn__1 - charts/global_optimality_gap",
             "MinAtar/Asterix-v0__dqn__2 - charts/global_optimality_gap",
             "MinAtar/Asterix-v0__dqn__3 - charts/global_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='DQN'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label})
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    # #####################
    # ##### w/ PPO ######
    # #####################
    
    keys_ = ["MinAtar/Asterix-v0__ppo__1 - charts/global_optimality_gap",
             "MinAtar/Asterix-v0__ppo__2 - charts/global_optimality_gap",
             "MinAtar/Asterix-v0__ppo__3 - charts/global_optimality_gap"]
    plot_data = get_data_frame(df, keys=keys_, res=res)

    label='PPO'
    plot_data = plot_data.rename(columns={0: 'Steps', 1: label} )
    sns.lineplot(data=plot_data, x='Steps', y=label, ax=ax3, label=label)
    # ax3.lines[-1].set_linestyle(linestyle[label])
    
    
    ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    ax3.set(ylabel='Global Optimality Gap')
    ax3.set(xlabel='Steps')
    ax3.legend()
    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("data/"+title+".svg")
    fig.savefig("data/"+title+".png")
    fig.savefig("data/"+title+".pdf")
