"""
Course: Machine Learning 2023/2024
Students :
Alberti Andrea	0622702370	a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355 c.attianese13@studenti.unisa.it
Capaldo Vincenzo 0622702347 v.capaldo7@studenti.unisa.it
Esposito Paolo 0622702292 p.esposito57@studenti.unisa.it
 """
import math
import numpy as np
"""
The CustomActionSpace class provides an interface for normalizing and denormalizing actions 
within a range specified by an array of bounds. The class contains two main methods: 
bound_actions, which normalizes the actions, and reverse_bound_actions, which denormalizes the actions
"""


class CustomActionSpace:
    def __init__(self):
        self.high = np.array([math.pi / 2, math.pi / 36])  # [90, 5] degree
        self.low = np.array([-math.pi / 2, -math.pi / 36])  # [-90,-5] degree

    def bound_actions(self, action):
        """
        Normalizes the actions to be into the bound.

        :param action:
        :return: normalized actions
        """
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.high - self.low)
        action += self.low
        return action

    def reverse_bound_actions(self, action):
        """
        Reverts the normalization

        :param action:
        :return: reverse normalized actions
        """
        action -= self.low
        action /= (self.high - self.low)
        action = action * 2 - 1
        return action
