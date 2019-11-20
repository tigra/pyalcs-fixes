import logging
from typing import Tuple

from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.action_planning.action_planning import \
    search_goal_sequence, suitable_cl_exists
from . import ClassifiersList, Configuration
from ...agents import Agent
from ...strategies.action_selection import choose_action

logger = logging.getLogger(__name__)


class ACS2(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList=None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        # import matplotlib.pyplot as plt
        # f=plt.figure(1)
        # ax=f.gca()
        # ax.scatter([0,19],[0,19], marker='.')
        # state_scatter = ax.scatter([0], [0])
        # import numpy as np
        # visits = np.zeros((20, 20))
        # visits_imshow = ax.imshow(visits)
        #
        # # imgs = self.create_action_implots()
        # plt.ion()


        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        action_set = ClassifiersList()
        done = False

        while not done:
            if self.cfg.do_action_planning and \
                    self._time_for_action_planning(steps + time):
                # Action Planning for increased model learning
                steps_ap, state, prev_state, action_set, \
                    action, last_reward = \
                    self._run_action_planning(env, steps + time, state,
                                              prev_state, action_set, action,
                                              last_reward)
                steps += steps_ap

            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,                     # rho ( == 0)
                    match_set.get_maximum_fitness(), # max predicted reward
                    self.cfg.beta,
                    self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        action_set,
                        state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                self.cfg.epsilon)
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)

            prev_state = state
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    ClassifiersList(),
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    0,
                    # last_reward,
                    self.cfg.beta,
                    self.cfg.gamma)
            if self.cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps,
                    self.population,
                    ClassifiersList(),
                    action_set,
                    state,
                    self.cfg.theta_ga,
                    self.cfg.mu,
                    self.cfg.chi,
                    self.cfg.theta_as,
                    self.cfg.do_subsumption,
                    self.cfg.theta_exp)

            steps += 1
            # env.render('human')

            # visits[int(state[0])-1, int(state[1])-1] += 1
            # state_scatter.set_offsets([int(state[1])-1, int(state[0])-1])
            # visits_imshow.set_data(visits)
            # visits_imshow.set_clim(visits.min(), visits.max())
            # plt.pause(0.001)

        if current_trial % 10 == 0:
            # print("Current trial: %d" % current_trial)
            # self.update_imgs(self.map_classifiers(self.population), 'explore', current_trial)
            self.update_fitness_bars(self.map_classifiers(self.population), 'explore', current_trial)

        # self.update_imgs(self.map_classifiers(self.population))

        # arrows = ['⬅', '➡', '⬆', '⬇']
        # def print_cl(cl):
        #     action = arrows[cl.action]
        #     print(f"{cl.condition} - {action} - {cl.effect} [fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}]")
        #
        # ax.imshow(visits)
        # ax.set_title("Steps: %d" % steps)
        # for cl in sorted(self.population, key=lambda c: -c.fitness)[:-10]:
        #     print_cl(cl)
        # print(len(self.population))

        # visits_imshow.set_clim(visits.min(), visits.max())
        # plt.pause(0.01)
        # plt.draw()

        return TrialMetrics(steps, last_reward)


    def map_classifiers(self, population: ClassifiersList):
        # classifiers = [cl for cl in classifiers if cl.does_anticipate_change()]
        # cs = [[cl for cl in classifiers if cl.action == a] for a in range(4)]

        import numpy as np
        total_fitnez = np.empty((4, 20, 20))
        total_fitnez[:] = np.nan
        for i in range(20):
            for j in range(20):
                state = Perception([str(i+1), str(j+1)])
                match_set = population.form_match_set(state)
                for a in range(4):
                    anticipated_change_cls = (cl for cl in match_set if cl.does_anticipate_change() and cl.action == a)
                    best_classifier = max(anticipated_change_cls, key=lambda cl: cl.fitness * cl.num, default=None)
                    if best_classifier is not None:
                        total_fitnez[a, i, j] = best_classifier.fitness * best_classifier.num
                    # if len(anticipated_change_cls) > 0:
                    #     best_classifier = max(anticipated_change_cls,
                    #                           key=lambda cl: cl.fitness * cl.num)
                    #     total_fitnez[a, i, j] = best_classifier.fitness*best_classifier.num
        return total_fitnez

    def create_fitness_bars(self):
        import numpy as np
        import matplotlib.pyplot as plt
        rng = np.arange(4)
        self.bars_figure = plt.figure(3)
        self.bar_axes = self.bars_figure.subplots(20, 20)
        self.bars=[[None]*20]*20
        for i in range(20):
            for j in range(20):
                self.bar_axes[i,j].set_axis_off()
                self.bars[i][j]=self.bar_axes[i, j].bar(rng, [100+i,100,100,200+j], color=['r','b','g','y'])
        self.bars_figure.tight_layout()


    def update_fitness_bars(self, fitnez, phase, trial):
        if not hasattr(self, 'bars') or self.bars is None:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        # c_min = np.nanmin(fitnez)
        # c_max = np.nanmax(fitnez)
        self.bars[5][5][0].set_height(300)
        for i in range(20):
            for j in range(20):
                for a in range(4):
                    f = fitnez[a, i, j]
                    self.bars[i][j][a].set_height(0 if f==np.NaN else f)
        self.bars_figure.suptitle('%s - trial %d' % (phase, trial))
        self.bars_figure.canvas.draw()
        plt.draw()
        plt.pause(0.01)


    def create_action_implots(self):
        # fitnez = self.map_classifiers(self.population)
        import numpy as np
        fitnez = np.zeros((4, 20, 20))
        import matplotlib.pyplot as plt
        self.imgs_figure = plt.figure(2)
        self.imgs_figure.suptitle('?')
        ax = [None]*6
        self.action_fitnesses = [None]*6
        ax[0]= self.imgs_figure.add_subplot(231, title='left')
        self.action_fitnesses[0] = ax[0].imshow(fitnez[0])

        ax[1]= self.imgs_figure.add_subplot(232, title='right')
        self.action_fitnesses[1] = ax[1].imshow(fitnez[1])

        ax[3]= self.imgs_figure.add_subplot(234, title='up')
        self.action_fitnesses[2] = ax[3].imshow(fitnez[2])

        ax[4]= self.imgs_figure.add_subplot(235, title='down')
        self.action_fitnesses[3] = ax[4].imshow(fitnez[3])

        ax[2]= self.imgs_figure.add_subplot(233, title='right>left')
        r = np.random.randint(0, 3, (20,20))
        # self.action_fitnesses[4] = ax[2].imshow(fitnez[0]<fitnez[1])
        self.action_fitnesses[4] = ax[2].imshow(r)

        ax[5]= self.imgs_figure.add_subplot(236, title='up>down')
        # self.action_fitnesses[5] = ax[5].imshow(fitnez[2]>fitnez[3])
        self.action_fitnesses[5] = ax[5].imshow(r)

        plt.pause(0.0001)
        return self.action_fitnesses

    def update_imgs(self, fitnez, phase='?', trial=-1):
        if not hasattr(self, 'imgs_figure') or self.imgs_figure is None:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        c_min = np.nanmin(fitnez)
        c_max = np.nanmax(fitnez)
        for i in range(4):
            self.action_fitnesses[i].set_data(fitnez[i])
            self.action_fitnesses[i].set_clim(c_min, c_max)

        self.action_fitnesses[4].set_data((fitnez[1]>fitnez[0]).astype(np.int) + (fitnez[1]>=fitnez[0]).astype(np.int))
        self.action_fitnesses[5].set_data((fitnez[2]>fitnez[3]).astype(np.int) + (fitnez[2]>=fitnez[3]).astype(np.int))
        self.action_fitnesses[4].set_clim(0, 2)
        self.action_fitnesses[5].set_clim(0, 2)
        # plt.draw()
        self.imgs_figure.suptitle('%s - trial %d' % (phase, trial))
        plt.pause(0.0001)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        # import matplotlib.pyplot as plt
        # f=plt.figure(3)
        # ax=f.gca()
        # ax.scatter([0,19],[0,19], marker='.')
        # state_scatter = ax.scatter([0], [0])
        # plt.ion()

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)

        last_reward = 0
        action_set = ClassifiersList()
        done = False

        actions = []
        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    # match_set.get_maximum_fitness()/4,
                    # 0,
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                epsilon=0.0)
            actions.append(action)
            iaction = self.cfg.environment_adapter.to_env_action(action)
            action_set = match_set.form_action_set(action)

            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            # state_scatter.set_offsets([int(state[1])-1, int(state[0])-1])
            # visits_imshow.set_data(visits)
            # visits_imshow.set_clim(visits.min(), visits.max())
            # plt.pause(0.00001)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1
            # print("Steps: %d" % steps)

        # print(len(actions), ''.join(map(str, actions)))
        if current_trial % 25 == 0:
            # self.update_imgs(self.map_classifiers(self.population), 'exploit', current_trial)
            self.update_fitness_bars(self.map_classifiers(self.population), 'exploit', current_trial)

        return TrialMetrics(steps, last_reward)

    def _run_action_planning(self,
                             env,
                             time: int,
                             state: Perception,
                             prev_state: Perception,
                             action_set: ClassifiersList,
                             action: int,
                             last_reward: int) -> Tuple[int, Perception,
                                                        Perception,
                                                        ClassifiersList,
                                                        int, int]:
        """
        Executes action planning for model learning speed up.
        Method requests goals from 'goal generator' provided by
        the environment. If goal is provided, ACS2 searches for
        a goal sequence in the current model (only the reliable classifiers).
        This is done as long as goals are provided and ACS2 finds a sequence
        and successfully reaches the goal.

        Parameters
        ----------
        env
        time
        state
        prev_state
        action_set
        action
        last_reward

        Returns
        -------
        steps
        state
        prev_state
        action_set
        action
        last_reward

        """
        logging.debug("** Running action planning **")

        if not hasattr(env.env, "get_goal_state"):
            logging.debug("Action planning stopped - "
                          "no function get_goal_state in env")
            return 0, state, prev_state, action_set, action, last_reward

        steps = 0
        done = False

        while not done:
            goal_situation = self.cfg.environment_adapter.to_genotype(
                env.env.get_goal_state())

            if goal_situation is None:
                break

            act_sequence = search_goal_sequence(self.population, state,
                                                goal_situation)

            # Execute the found sequence and learn during executing
            i = 0
            for act in act_sequence:
                if act == -1:
                    break

                match_set = self.population.form_match_set(state)

                if action_set is not None and len(prev_state) != 0:
                    ClassifiersList.apply_alp(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        action,
                        state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                    ClassifiersList.apply_reinforcement_learning(
                        action_set,
                        last_reward,
                        0,
                        self.cfg.beta,
                        self.cfg.gamma)
                    if self.cfg.do_ga:
                        ClassifiersList.apply_ga(
                            time + steps,
                            self.population,
                            match_set,
                            action_set,
                            state,
                            self.cfg.theta_ga,
                            self.cfg.mu,
                            self.cfg.chi,
                            self.cfg.theta_as,
                            self.cfg.do_subsumption,
                            self.cfg.theta_exp)

                action = act
                action_set = ClassifiersList.form_action_set(match_set, action)

                iaction = self.cfg.environment_adapter.to_lcs_action(action)

                raw_state, last_reward, done, _ = env.step(iaction)
                prev_state = state

                state = self.cfg.environment_adapter.to_genotype(raw_state)

                if not suitable_cl_exists(action_set, prev_state,
                                          action, state):

                    # no reliable classifier was able to anticipate
                    # such a change
                    break

                steps += 1
                i += 1

            if i == 0:
                break

        return steps, state, prev_state, action_set, action, last_reward

    def _time_for_action_planning(self, time):
        return time % self.cfg.action_planning_frequency == 0
