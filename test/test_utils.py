import numpy as np
import unittest
from utils import math as um

class TestUtils(unittest.TestCase):

    def test_tuple_to_idx1(self):
        ds = (3,3,3)
        tp = (1,0,2)
        idx = um.tuple_to_idx(tp,ds)
        tp2 = um.idx_to_tuple(idx,ds)

        self.assertEqual(tp,tp2,"%s -> %d -> %s" % (str(tp),idx,str(tp2)))

    def test_tuple_to_idx2(self):
        ds = (2,8,7)
        tp = (0,3,4)
        idx = um.tuple_to_idx(tp,ds)
        tp2 = um.idx_to_tuple(idx,ds)

        self.assertEqual(tp,tp2,"%s -> %d -> %s" % (str(tp),idx,str(tp2)))

    def test_tuple_to_idx3(self):
        ds = (2,8,7)
        tp = (1,6,2)
        idx = um.tuple_to_idx(tp,ds)
        tp2 = um.idx_to_tuple(idx,ds)

        self.assertEqual(tp,tp2,"%s -> %d -> %s" % (str(tp),idx,str(tp2)))
