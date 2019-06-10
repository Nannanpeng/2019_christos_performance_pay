import numpy as np
import unittest
from utils import math as um

class TestUtils(unittest.TestCase):

    def test_tuple_to_idx1(self):
        ds = (3,3,3)
        tp = (1,0,2)
        idx = um.tuple_to_idx(tp,ds)
        tp2 = um.idx_to_tuple(idx,ds)

        self.assertEqual(tp,tp2,"%s should equal %s" % (str(tp),str(tp2)))
