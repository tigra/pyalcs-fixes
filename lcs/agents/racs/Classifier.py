from __future__ import annotations

import random
from typing import Optional, List, Callable

from lcs import Perception
from lcs.representations import UBR
from . import Condition, Effect, Mark, Configuration


class Classifier:

    def __init__(self,
                 condition: Optional[Condition]=None,
                 action: Optional[int]=None,
                 effect: Optional[Effect]=None,
                 quality: float=0.5,
                 reward: float = 0.5,
                 intermediate_reward: float = 0.0,
                 experience: int=1,
                 talp=None,
                 tga: int = 0,
                 tav: float=0.0,
                 cfg: Optional[Configuration]=None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_condition(initial):
            if initial:
                return Condition(initial, cfg=cfg)

            return Condition.generic(cfg=cfg)

        def build_effect(initial):
            if initial:
                return Effect(initial, cfg=cfg)

            return Effect.pass_through(cfg=cfg)

        self.condition = build_condition(condition)
        self.action = action
        self.effect = build_effect(effect)

        self.mark = Mark(cfg=cfg)
        self.q = quality
        self.r = reward
        self.ir = intermediate_reward

        self.exp = experience
        self.talp = talp
        self.tga = tga
        self.tav = tav

    @classmethod
    def copy_from(cls, old_cls: Classifier, time: int):
        """
        Copies old classifier with given time (tga, talp).
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
        old_cls: Classifier
            classifier to copy from
        time: int
            time of creation / current epoch

        Returns
        -------
        Classifier
            copied classifier
        """
        new_cls = cls(
            condition=Condition(old_cls.condition, old_cls.cfg),
            action=old_cls.action,
            effect=old_cls.effect,
            quality=old_cls.q,
            reward=old_cls.r,
            intermediate_reward=old_cls.ir,
            cfg=old_cls.cfg)

        new_cls.tga = time
        new_cls.talp = time
        new_cls.tav = old_cls.tav

        return new_cls

    @property
    def specified_unchanging_attributes(self) -> List[int]:
        """
        Determines the number of specified unchanging attributes in
        the classifier. An unchanging attribute is one that is anticipated
        not to change in the effect part.

        Returns
        -------
        List[int]
            list specified unchanging attributes indices
        """
        indices = []

        for idx, (cpi, epi) in enumerate(zip(self.condition, self.effect)):
            if cpi != self.cfg.classifier_wildcard and \
                    epi == self.cfg.classifier_wildcard:
                indices.append(idx)

        return indices

    def specialize(self,
                   previous_situation: Perception,
                   situation: Perception) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1 and returns a condition which specifies
        the attributes which must be specified in the condition part.
        The specific attributes in the returned conditions are set to
        the necessary values.

        For real-valued representation a narrow, fixed point UBR is created
        for condition and effect part using the encoded perceptions.

        Parameters
        ----------
        previous_situation: Perception
            previous raw perception obtained from environment
        situation: Perception
            current raw perception obtained from environment

        Returns
        -------
        """
        p0_enc = list(map(self.cfg.encoder.encode, previous_situation))
        p1_enc = list(map(self.cfg.encoder.encode, situation))

        for idx, item in enumerate(p1_enc):
            if p0_enc[idx] != p1_enc[idx]:
                self.effect[idx] = UBR(p1_enc[idx], p1_enc[idx])
                self.condition[idx] = UBR(p0_enc[idx], p0_enc[idx])

    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp

    def increase_quality(self) -> float:
        self.q += self.cfg.beta * (1 - self.q)
        return self.q

    def decrease_quality(self) -> float:
        self.q -= self.cfg.beta * self.q
        return self.q

    def does_anticipate_correctly(self,
                                  previous_situation: Perception,
                                  situation: Perception) -> bool:
        """
        Checks whether classifier correctly performs anticipation.

        Parameters
        ----------
        previous_situation: Perception
            Previously observed perception
        situation: Perception
            Current perception

        Returns
        -------
        bool
            True if anticipation is correct, False otherwise
        """
        p0_enc = list(map(self.cfg.encoder.encode, previous_situation))
        p1_enc = list(map(self.cfg.encoder.encode, situation))

        # FIXME: Think - in this proposition there is no idea of an wildcard.
        # However it works fine. Will see later. There might be a problem
        # that for very wide Effect (wildcard) everything will be accepted as
        # correctly anticipated. So some specialization pressure needs to be
        # applied

        for idx, eitem in enumerate(self.effect):
            if p1_enc[idx] in eitem:
                if p0_enc[idx] == p1_enc[idx]:
                    pass
            else:
                return False

        return True

    def set_mark(self, perception: Perception) -> None:
        """
        Marks classifier with given perception taking into consideration its
        condition.

        Specializes the mark in all attributes which are not specified
        in the conditions, yet

        Parameters
        ----------
        perception: Perception
            current situation
        """
        if self.mark.set_mark_using_condition(self.condition, perception):
            self.ee = 0

    def set_alp_timestamp(self, time: int) -> None:
        """
        Sets the ALP time stamp and the application average parameter.

        Parameters
        ----------
        time: int
            current time step
        """
        # TODO p5: write test
        if 1. / self.exp > self.cfg.beta:
            self.tav = (self.tav * self.exp + (time - self.talp)) / (
                self.exp + 1)
        else:
            self.tav += self.cfg.beta * ((time - self.talp) - self.tav)

        self.talp = time

    def is_marked(self):
        return self.mark.is_marked()

    def generalize_unchanging_condition_attribute(
            self, randomfunc: Callable=random.choice) -> bool:
        """
        Generalizes one randomly unchanging attribute in the condition.
        An unchanging attribute is one that is anticipated not to change
        in the effect part.

        Parameters
        ----------
        randomfunc: Callable
            function returning attribute index to generalize
        Returns
        -------
        bool
            True if attribute was generalized, False otherwise
        """
        if len(self.specified_unchanging_attributes) > 0:
            ridx = randomfunc(self.specified_unchanging_attributes)
            self.condition.generalize(ridx)
            return True

        return False
