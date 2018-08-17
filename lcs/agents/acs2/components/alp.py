from random import random, randint
from typing import Optional

from lcs import Perception
from lcs.agents.acs2 import Classifier, Configuration


def cover(previous_situation: Perception,
          action: int,
          situation: Perception,
          time: int,
          cfg: Configuration) -> Classifier:
    """
    Covering - creates a classifier that anticipates a change correctly.

    :param previous_situation:
    :param action:
    :param situation:
    :param time: current epoch
    :param cfg: configuration

    :return: new classifier
    """
    new_cl = Classifier(action=action, cfg=cfg)
    # TODO: p5 exp=0, r=0 (paper)
    new_cl.tga = time
    new_cl.talp = time

    new_cl.specialize(previous_situation, situation)

    return new_cl


def expected_case(cl: Classifier,
                  perception: Perception,
                  time: int) -> Optional[Classifier]:
    """
    Controls the expected case of a classifier. If the classifier
    is too specific it tries to add some randomness to it by
    generalizing some attributes.

    :param cl:
    :param perception:
    :param time:
    :return: new classifier or None
    """
    diff = cl.mark.get_differences(perception)

    if diff.specificity == 0:
        cl.increase_quality()
        return None

    no_spec = len(cl.specified_unchanging_attributes)
    no_spec_new = diff.specificity
    child = cl.copy_from(cl, time)

    if no_spec >= cl.cfg.u_max:
        while no_spec >= cl.cfg.u_max:
            res = cl.generalize_unchanging_condition_attribute()
            assert res is True
            no_spec -= 1

        while no_spec + no_spec_new > cl.cfg.u_max:
            if random() < 0.5:
                diff.generalize_specific_attribute_randomly()
                no_spec_new -= 1
            else:
                if cl.generalize_unchanging_condition_attribute():
                    no_spec -= 1
    else:
        while no_spec + no_spec_new > cl.cfg.u_max:
            diff.generalize_specific_attribute_randomly()
            no_spec_new -= 1

    child.condition.specialize(new_condition=diff)

    if child.q < 0.5:
        child.q = 0.5

    return child


def unexpected_case(cl: Classifier,
                    previous_perception: Perception,
                    perception: Perception,
                    time: int) -> Optional[Classifier]:
    """
    Controls the unexpected case of the classifier.

    :param cl:
    :param previous_perception:
    :param perception:
    :param time:
    :return: specialized classifier if generation was possible,
    None otherwise
    """
    cl.decrease_quality()
    cl.set_mark(previous_perception)

    # Return if the effect is not specializable
    if not cl.effect.is_specializable(previous_perception, perception):
        return None

    child = cl.copy_from(cl, time)

    # TODO: p5 maybe also take into consideration cl.E = # (paper)
    child.specialize(previous_perception, perception)

    if child.q < 0.5:
        child.q = 0.5

    return child
