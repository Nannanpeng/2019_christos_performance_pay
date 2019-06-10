import numpy as np
import unittest
import math
from solver.models import State
from solver.algorithm import Multi_LI_VFA


class TestValueFunction(unittest.TestCase):
    def setUp(self):
        ds = (2,2)
        T = 4
        grid = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        self.vfa = Multi_LI_VFA(T,ds,grid)

    def test_vfa_raises1(self):
        state = State(4,(1,0),(0.5,0.5))
        self.assertRaises(RuntimeError, self.vfa, state)

    def test_vfa_raises2(self):
        state = State(4,(1,0),(0.5,0.5))
        vals = np.array([0.5,0.5,0.5,0.5])
        self.vfa.add_values(state.t,state.discrete,vals)
        self.assertRaises(RuntimeError, self.vfa.add_values, state.t, state.discrete, vals)

    def test_vfa_predict(self):
        state = State(4,(1,0),(0.5,0.5))
        vals = np.array([0.5,0.5,0.5,0.5])
        self.vfa.add_values(state.t, state.discrete, vals)
        estimate = self.vfa(state)
        self.assertEqual(estimate,0.5)

    def test_vfa_predict_oob(self):
        state = State(4,(1,0),(-1.0,0.5))
        vals = np.array([0.5,0.5,0.5,0.5])
        self.vfa.add_values(state.t, state.discrete, vals)
        estimate = self.vfa(state)
        self.assertEqual(0.5,estimate)
