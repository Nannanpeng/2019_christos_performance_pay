import numpy as np
import unittest
from solver.algorithm import VFA_LI

class TestValueFunction(unittest.TestCase):
    def setUp(self):
        ds = (2,2)
        T = 4
        grid = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        self.vfa = VFA_LI(T,ds,grid)

    def test_vfa1(self):

        self.assertEqual(0,0)
