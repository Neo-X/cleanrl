


## A special buffer to help track the optimality gap between data generated
from typing import Any, Dict, Generator, List, Optional, Union
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces
import numpy as np
import torch as th

class BufferGap(ReplayBuffer):
    """
    A special buffer to help track the optimality gap between data generated
    by the agent and the optimal policy.
    """

    def __init__(self, 
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        *args, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,
                         optimize_memory_usage, handle_timeout_termination)
        # super().__init__(*args, **kwargs)
        self.gap = 0.0
        self._gap_percentage = 0.05
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        ## Check is the info dictionary contains the return key "r" if it does add that to the returns vector
        
        self.returns[self.pos] = np.array([info.get("return") for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)

import heapq, collections, torch, gym

class BufferGapV2():

    def __init__(self, 
        buffer_size: int,
        top_buffer_percet: float = 0.05,
        policy = False, 
        device = "auto", 
        args = None,
        envs = None,
    ):
        
        self._max_return = -100000
    ## max_returns is a list of the top 10 episodic returns
        self._max_returns = []
        self._max_return_buff_size = buffer_size * (top_buffer_percet)
        self._top_buffer_percet = top_buffer_percet
        ## returns is a deque of the last 100 episodic returns
        self._returns = collections.deque(maxlen=buffer_size)

        self._policy = policy
        self._device = device

        # env setup
        self._envs = envs
        self._args = args
        self._last_eval = 0

        self._best_traj = []


    def add(self, info):
        return_ = info.get("r")
        self._returns.append(return_)
    
        if return_ > self._max_return:
            self._max_return = return_
            ## Store the best trajectory
            self._best_traj = info["actions"]
        
        if len(self._max_returns) == 0:
            ## If this is the first return jsut story that return
            self._max_returns = [return_]
            # heapq.heapify(max_returns)
        if len(self._max_returns) > 0 and return_ > min(self._max_returns):
            ## Repalce the minimum value in max_returns with the new episodic return if the buffer is full
            if (len(self._max_returns) < self._max_return_buff_size ):
                ## If the buffer is not full, just append the new episodic return
                self._max_returns.append(return_)
                heapq.heapify(self._max_returns)
            else:
                heapq.heapreplace(self._max_returns, return_)


    def plot_gap(self, writer, step: int):
        """
        Plot the gap between the current return and the maximum return
        """
        returns_ = list(self._returns)
        heapq.heapify(returns_)
        writer.add_scalar("charts/best_trajectory_return", self._max_return, step)
        writer.add_scalar("charts/avg_top_returns_global", np.mean(list(self._max_returns)), step)
        writer.add_scalar("charts/avg_top_returns_local", np.mean(heapq.nlargest(max(int(self._top_buffer_percet * len(returns_)), 1), returns_)), step)
        writer.add_scalar("charts/global_optimality_gap", np.mean(list(self._max_returns)) - np.mean(returns_), step)
        writer.add_scalar("charts/local_optimality_gap", np.mean(heapq.nlargest(max(int(self._top_buffer_percet * len(returns_)), 1), returns_)) - np.mean(returns_), step)
        
        ## Get performance for the deterministic policy
        if step - self._last_eval > 10000:
            returns = self.eval_deterministic()
            self._last_eval = step
            writer.add_scalar("charts/deterministic_returns", np.mean(returns), step)
            returns = self.eval_deterministic(best=True)
            writer.add_scalar("charts/replay_best_returns", np.mean(returns), step)


    def eval_deterministic(self, best=False) -> np.ndarray:
        """
        Evaluate the policy deterministically
        """
        obs, _ = self._envs.reset(seed=self._args.seed)
        # q_values = self._policy(torch.Tensor(obs).to(self._device))
        # actions = torch.argmax(q_values, dim=1).cpu().numpy()
        returns = []
        # if best==True: max_t = len(self._best_traj) else: max_t = 1000
        ## Use spec.max_episode_steps to make sure there is a return in the info dict
        max_t = len(self._best_traj) if best==True else self._envs.envs[0].spec.max_episode_steps
        max_t = 1000 if max_t==None else max_t
        samples_ = 1
        for j in range(samples_):
            # obs, _ = self._envs.reset()
            returns_ = np.zeros(self._envs.num_envs, dtype=np.float32)
            for t in range(max_t):
                
                actions = [self._best_traj[t] for _ in range(self._envs.num_envs)] if best==True else self._policy.get_action_deterministic(torch.Tensor(obs).to(self._device)).cpu().numpy()
                
                obs, reward, terminations, truncations, infos = self._envs.step(actions)
                ## At the end of an episode the info disctionary has other junk in it.
                if "final_info" in infos: ## This final info logic is not very clear
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            returns.extend(info['episode']['r'])
                            break
                    break 
                            # pass
                else:
                    returns_ += infos['reward']
                # Check if the episode is done
                if terminations.any() or truncations.any():
                    dones = np.logical_or(terminations, truncations)
                    returns.extend(np.where(dones, returns_, 0.0))
                    break
                elif t == (max_t - 1): ## If the code makes it to here then no episode finished early
                    returns.extend(returns_)
                    break
            # print("COde shoud not get here")
        # assert(len(returns) == samples_), f"Returns length is {len(returns)} while expected {samples_}"
        return np.mean(returns)

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminations, truncations, infos = self._envs.step(actions)
        # return_ += rewards
        # infos["return"] = return_

        

"""Wrapper that tracks the cumulative rewards and episode lengths."""
import time
from collections import deque
from typing import Optional

import numpy as np

import gymnasium as gym


class RecordEpisodeStatisticsV2(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.episode_actions = []
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_actions = []
        # self.episode_actions = np.zeros(self.num_envs, dtype=np.float32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.episode_actions.append(action)
        # infos['reward'] = rewards
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                        ),
                    "actions": np.where(dones, self.episode_actions, 0) if isinstance(self.env.action_space, gym.spaces.Discrete) else np.where(dones, self.episode_actions, 0),
                }
                # print("Actions", np.where(dones, self.episode_actions, 0))
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        infos['reward']= rewards
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


"""A synchronous vector environment."""
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv


__all__ = ["SyncVectorEnv"]


class SyncVectorEnvV2(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None
        self._seed = 0

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)
        self._seed = seed

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                # env.seed(self._seed)
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True
