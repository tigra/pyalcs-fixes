import logging
import random
from itertools import groupby

import numpy as np

logger = logging.getLogger(__name__)


def choose_action(cll, all_actions: int, epsilon: float) -> int:
    """
    Chooses which action to execute given classifier list (match set).

    Parameters
    ----------
    cll:
        list of classifiers
    all_actions:
        int: number of all possible actions available
    epsilon: float
        Probability of executing exploration path

    Returns
    -------
    int
        number of chosen action
    """
    if random.random() < epsilon:
        logger.debug("\t\tExploration path")
        return explore(cll, all_actions)

    logger.debug("\t\tExploitation path")
    return exploit(cll, all_actions)


def explore(cll, all_actions: int, pb: float = 0.5) -> int:
    """
    Chooses action according to current exploration policy

    Parameters
    ----------
    cll:
        list of classifiers
    all_actions: int
        number of all possible actions available
    pb: float
        probability of biased exploration

    Returns
    -------
    int
        action to be executed
    """
    if random.random() < pb:
        # We are in the biased exploration
        if random.random() < 0.5:
            return choose_latest_action(cll, all_actions)
        else:
            return choose_action_from_knowledge_array(cll, all_actions)

    return choose_random_action(all_actions)


def exploit(cll, all_actions: int) -> int:
    """
    Chooses the best action using deterministic action voting.

    All classifiers anticipating change will be compared.
    Best action will be selected from classifier having the maximum
    `fitness` x `numerosity` value

    If there is no classifier in list (or none is predicting change)
    then a random action is returned.

    Parameters
    ----------
    cll:
        list of classifiers
    all_actions: int
        number of all possible actions available

    Returns
    -------
    int
        action from the best classifier
    """

    def dump_cls(cls):
        return sorted([(cl.condition, cl.action, cl.effect, cl.fitness, cl.num, cl.fitness*cl.num) for cl in cls],
                      key=lambda t: t[-1])

    best_classifier = None
    anticipated_change_cls = [cl for cl in cll
                              if cl.does_anticipate_change()]
    # print(dump_cls(anticipated_change_cls))

    if len(anticipated_change_cls) > 0:
        best_classifier = max(anticipated_change_cls,
                              key=lambda cl: cl.fitness * cl.num)
        # print(best_classifier)

    if best_classifier is not None:
        return best_classifier.action

    random_action = choose_random_action(all_actions)
    print("random: %s" % random_action)
    return random_action


def choose_latest_action(cll, all_actions: int) -> int:
    """
    Chooses latest executed action ("action delay bias")

    Parameters
    ----------
    cll:
        list of classifiers
    all_actions: int
        number of all possible actions available

    Returns
    -------
    int
        chosen action number
    """
    last_executed_cls = None
    number_of_cls_per_action = {i: 0 for i in range(all_actions)}

    if len(cll) > 0:
        last_executed_cls = min(cll, key=lambda cl: cl.talp)

        cll.sort(key=lambda cl: cl.action)
        for _action, _clss in groupby(cll, lambda cl: cl.action):
            number_of_cls_per_action[_action] = \
                sum([cl.num for cl in _clss])

    # If there are some actions with no classifiers - select them
    for action, nCls in number_of_cls_per_action.items():
        if nCls == 0:
            return action

    # Otherwise return the action of the last executed classifier
    if last_executed_cls:
        return last_executed_cls.action

    # if there is no classifiers - select random action
    return choose_random_action(all_actions)


def choose_action_from_knowledge_array(cll, all_actions: int) -> int:
    """
    Creates 'knowledge array' that represents the average quality of the
    anticipation for each action in the current list. Chosen is
    the action, ACS2 knows least about the consequences.

    Parameters
    ----------
    cll:
        list of classifiers
    all_actions: int
        number of all possible actions available

    Returns
    -------
    int
        chosen action
    """
    knowledge_array = {i: 0.0 for i in range(all_actions)}

    cll.sort(key=lambda cl: cl.action)

    for _action, _clss in groupby(cll, lambda cl: cl.action):
        _classifiers = [cl for cl in _clss]

        agg_q = sum(cl.q * cl.num for cl in _classifiers)
        agg_num = sum(cl.num for cl in _classifiers)

        knowledge_array[_action] = agg_q / float(agg_num)

    by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
    with_best_quality = [ki for ki in by_quality if ki[1] == by_quality[0][1]]
    # action = by_quality[0][0]
    action = random.choice([ki[0] for ki in with_best_quality])  # Or else there is a bias towards Left, Right, Up

    return action


def choose_random_action(all_actions: int) -> int:
    """
    Chooses one of the possible actions in the environment randomly

    Parameters
    ----------
    all_actions: int
        number of all possible actions available

    Returns
    -------
    int
        random action number
    """
    return np.random.randint(all_actions)
