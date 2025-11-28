# from rliable import library as rly
# from rliable import metrics
# from rliable import plot_utils
import numpy as np
import pandas as pd

# Import specific modules
from rlplot import metrics
from rlplot import library as rly
from rlplot import plot_utils
from rlplot.plot_helpers import \
    load_all_exp_data, generate_intervals, generate_pairs, \
    save_metric, get_metric, read_and_norm_algo_scores, save_fig, \
    get_rank_matrix


# from make_plots import *
REPS = 1000
CONFIDENCE = 0.68


# Example performance data
# Each algorithm has scores for multiple runs across multiple tasks
# Shape: (num_runs, num_tasks)
# from https://gist.github.com/vwxyzjn/eb84ea5386cababf3571aabfb8f4474c
atari_human_normalized_scores = {
    'Alien-v5': (227.8, 7127.7),
    'Amidar-v5': (5.8, 1719.5),
    'Assault-v5': (222.4, 742.0),
    'Asterix-v5': (210.0, 8503.3),
    'Asteroids-v5': (719.1, 47388.7),
    'Atlantis-v5': (12850.0, 29028.1),
    'BankHeist-v5': (14.2, 753.1),
    'BattleZone-v5': (2360.0, 37187.5),
    'BeamRider-v5': (363.9, 16926.5),
    'Berzerk-v5': (123.7, 2630.4),
    'Bowling-v5': (23.1, 160.7),
    'Boxing-v5': (0.1, 12.1),
    'Breakout-v5': (1.7, 30.5),
    'Centipede-v5': (2090.9, 12017.0),
    'ChopperCommand-v5': (811.0, 7387.8),
    'CrazyClimber-v5': (10780.5, 35829.4),
    'Defender-v5': (2874.5, 18688.9),
    'DemonAttack-v5': (152.1, 1971.0),
    'DoubleDunk-v5': (-18.6, -16.4),
    'Enduro-v5': (0.0, 860.5),
    'FishingDerby-v5': (-91.7, -38.7),
    'Freeway-v5': (0.0, 29.6),
    'Frostbite-v5': (65.2, 4334.7),
    'Gopher-v5': (257.6, 2412.5),
    'Gravitar-v5': (173.0, 3351.4),
    'Hero-v5': (1027.0, 30826.4),
    'IceHockey-v5': (-11.2, 0.9),
    'Jamesbond-v5': (29.0, 302.8),
    'Kangaroo-v5': (52.0, 3035.0),
    'Krull-v5': (1598.0, 2665.5),
    'KungFuMaster-v5': (258.5, 22736.3),
    'MontezumaRevenge-v5': (0.0, 4753.3),
    'MsPacman-v5': (307.3, 6951.6),
    'NameThisGame-v5': (2292.3, 8049.0),
    'Phoenix-v5': (761.4, 7242.6),
    'Pitfall-v5': (-229.4, 6463.7),
    'Pong-v5': (-20.7, 14.6),
    'PrivateEye-v5': (24.9, 69571.3),
    'Qbert-v5': (163.9, 13455.0),
    'Riverraid-v5': (1338.5, 17118.0),
    'RoadRunner-v5': (11.5, 7845.0),
    'Robotank-v5': (2.2, 11.9),
    'Seaquest-v5': (68.4, 42054.7),
    'Skiing-v5': (-17098.1, -4336.9),
    'Solaris-v5': (1236.3, 12326.7),
    'SpaceInvaders-v5': (148.0, 1668.7),
    'StarGunner-v5': (664.0, 10250.0),
    'Surround-v5': (-10.0, 6.5),
    'Tennis-v5': (-23.8, -8.3),
    'TimePilot-v5': (3568.0, 5229.2),
    'Tutankham-v5': (11.4, 167.6),
    'UpNDown-v5': (533.4, 11693.2),
    'Venture-v5': (0.0, 1187.5),
    # Note the random agent score on Video Pinball is sometimes greater than the
    # human score under other evaluation methods.
    'VideoPinball-v5': (16256.9, 17667.9),
    'WizardOfWor-v5': (563.5, 4756.5),
    'YarsRevenge-v5': (3092.9, 54576.9),
    'Zaxxon-v5': (32.5, 9173.3),
}

task_list = ["Asterix-v5", "BattleZone-v5", "MontezumaRevenge-v5", "NameThisGame-v5", "SpaceInvaders-v5"]

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
    min_ = []
    for i in range(len(jobs)): 
        data_ = df[jobs[i]+key][:max].to_numpy()
        data_ = deNan(data_)
        print(jobs[i]+key, data_)

        # plot_data.extend([(step_, val, std_) for step_, val, std_ in zip(steps_, data_, stds_)])
        print("key", jobs[i]+key, "data_:", data_)
        plot_data.append(np.mean(data_))
        min_.append(np.min(data_))  
    
    return (np.array(plot_data), np.array(min_))

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


    aggregate_names = ['IQM', 'Mean', 'Median', 'Global Optimality Gap']
    # Get interval estimates for IQM (Interquartile Mean)
    aggregate_func_mapper = {
        'Mean': metrics.aggregate_mean,
        'IQM': metrics.aggregate_iqm,
        'Median': metrics.aggregate_median,
        'Global Optimality Gap': metrics.aggregate_optimality_gap,
    }
    def aggregate_func(x): return \
        np.array([aggregate_func_mapper[name](x) for name in aggregate_names])
    
    scores = {}
    scores['SAC'] = []
    datadirs = [
        './data/SAC_on_MinAtar_SpaceInvaders.csv',
        './data/SAC_on_MinAtar_Breakout.csv',
        # './data/DQN_BattleZone.csv',
        # './data/DQN_NameThisGame_ResNet.csv',
        # './data/DQN_MR-all.csv',
    ]
    for i in range(len(datadirs)):
        df = pd.read_csv(datadirs[i])
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
        scores['SAC'].append(((scores_ - min_)/((scores_max+0.0001) - min_))) ## For some envs no reward is ever found.

    scores['SAC+RND'] = []
    datadirs = [
        './data/SAC_with_RND_on_MinAtar_SpaceInvaders.csv',
        './data/SAC_with_RND_on_MinAtar_Breakout.csv',
        # './data/PPO_NameThisGame.csv',
        # './data/PPO_SpaceInvaders.csv',
        # './data/PPO_MinAtar_SpaceInvaders.csv',
        # './data/PPO_HalfCheetah_4_layers.csv',
    ]
    for i in range(len(datadirs)):
        df = pd.read_csv(datadirs[i])
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
        scores['SAC+RND'].append(((scores_ - min_)/((scores_max+0.0001) - min_)))

    
    scores['PQN'] = []
    datadirs = [
        './data/PQN_on_MinAtar_SpaceInvaders.csv',
        './data/PQN_on_MinAtar_Breakout.csv',
        './data/PQN_on_ALE_NameThisGame.csv',
        './data/PQN_on_ALE_BattleZone.csv',
        # './data/PPO_NameThisGame.csv',
        # './data/PPO_SpaceInvaders.csv',
        # './data/PPO_MinAtar_SpaceInvaders.csv',
        # './data/PPO_HalfCheetah_4_layers.csv',
    ]
    for i in range(len(datadirs)):
        df = pd.read_csv(datadirs[i])
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
        scores['PQN'].append(((scores_ - min_)/((scores_max+0.0001) - min_)))

    scores['PQN+RND'] = []
    datadirs = [
        './data/PQN_with_RND_on_MinAtar_SpaceInvaders.csv',
        './data/PQN_with_RND_on_MinAtar_Breakout.csv',
        './data/PQN_with_RND_on_ALE_NameThisGame.csv',
        './data/PQN_with_RND_on_ALE_BattleZone.csv',
        # './data/PPO_SpaceInvaders.csv',
        # './data/PPO_MinAtar_SpaceInvaders.csv',
        # './data/PPO_HalfCheetah_4_layers.csv',
    ]
    for i in range(len(datadirs)):
        df = pd.read_csv(datadirs[i])
        jobs = get_jobs(df)
        (scores_, min_) = get_data_frame(df, key=" - charts/episodic_return", jobs=jobs)
        (scores_max, min_) = get_data_frame(df, key=" - charts/avg_top_returns_global", jobs=jobs)
        scores['PQN+RND'].append(((scores_ - min_)/((scores_max+0.0001) - min_)))

    for key in scores.keys():
        min_len = min([len(scores[key][i]) for i in range(len(scores[key]))])
        scores[key] = np.array([ scores_[:min_len] for scores_ in scores[key] ])
    normalized_algo_scores = scores.copy()


    aggregate_scores, aggregate_score_cis = \
        rly.get_interval_estimates(
            normalized_algo_scores, aggregate_func,
            reps=REPS, confidence_interval_size=CONFIDENCE
        )
    # Plot the interval estimates for different metrics
    algos = list(scores.keys())
    # point_estimates = [iqm_scores, mean_scores, median_scores, optimal_scores]
    # interval_estimates = [iqm_cis, mean_cis, median_cis, optimal_cis]

    print("Aggregate Scores:", aggregate_scores)

    fig, axes = plot_utils.plot_metric_value(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=aggregate_names,
        algorithms=algos,
        milestone='1m',
        xlabel='Normalized Score',
        xlabel_y_coordinate=-0.65,
    )
    # plt.tight_layout()
    # plt.show()
    save_fig(fig, 'metric_value_gap_global_SAC', "data")

 