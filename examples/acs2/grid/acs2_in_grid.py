import os, sys
import logging
import time

sys.path.insert(0, os.path.abspath('../../../../openai-envs'))


import gym
# noinspection PyUnresolvedReferences
import gym_grid

from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

arrows = ['⬅', '➡', '⬆', '⬇']


def print_cl(cl):
    action = arrows[cl.action]
    print(f"{cl.condition} - {action} - {cl.effect} [fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}]")


def dump_classifiers(classifiers):
    for cl in sorted(classifiers, key=lambda c: -c.fitness)[:-20]:
        print_cl(cl)


if __name__ == '__main__':
    # Load desired environment
    grid = gym.make('grid-20-v0')
    grid._max_episode_steps = 2000
    grid.env.REWARD = 2000
    # grid.render()

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        # epsilon=0.7, # 1.0
        epsilon=0.95,
        beta=0.1,
        gamma=0.8,
        theta_exp=20,
        theta_ga=100,
        theta_as=20,
        theta_r=0.95,
        theta_i=0.2,
        do_ga=True,
        # do_ga=False,
        mu=0.02,
        u_max=2,
        metrics_trial_frequency=1)

    print(cfg)

    # Explore the environment
    logging.info("Exploring grid")
    agent = ACS2(cfg)
    # agent.create_action_implots()
    # agent.create_fitness_bars()

    explore_start = time.time()

    population, explore_metrics = agent.explore(grid, 1000)

    explore_duration = time.time() - explore_start
    print("Explore duration: %f" % explore_duration)

    print(explore_metrics)
    avg_explore_steps = sum([m['steps_in_trial'] for m in explore_metrics]) / len(explore_metrics)
    print("Average explore steps: %f" % avg_explore_steps)


    agent.update_imgs(agent.map_classifiers(agent.population))

    dump_classifiers(population)

    exploit_start = time.time()

    metrics = agent.exploit(grid, 1000)

    exploit_duration = time.time() - exploit_start
    print("Exploit duration: %f" % exploit_duration)

    dump_classifiers(agent.population)

    import matplotlib.pyplot as plt
    print(metrics)

    exploit_steps = [m['steps_in_trial'] for m in metrics[1]]
    avg_exploit_steps = sum(exploit_steps) / len(exploit_steps)
    print("Average exploit steps: %f" % avg_exploit_steps)
    agent.update_imgs(agent.map_classifiers(agent.population))
    f=plt.figure(4)
    import numpy as np
    plt.plot(np.arange(len(metrics[1])), exploit_steps)
    f.gca().axhline(22, ls=':', label='optimal')
    plt.legend()
    plt.show(block=True)
