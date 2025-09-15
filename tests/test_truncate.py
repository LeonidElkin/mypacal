import unittest
import numpy as np

from pacal import UniformDistr, ConstDistr, NormalDistr
from pacal.distr import TruncDistr
from pacal.rv import TruncRV, RV


class TestTruncate(unittest.TestCase):
    def test_trunc_distr(self):
        d = UniformDistr(0, 10)
        truncated = d.trunc(2, 8)
        self.assertEqual(truncated.range(), (2, 8))
        self.assertIsInstance(truncated, TruncDistr)

    def test_trunc_rv(self):
        base_rv = RV(sym="X", a=0, b=10)
        trunc_rv = TruncRV(base_rv, 3, 7)
        self.assertEqual(trunc_rv.getSegments(), (3, 7))
        self.assertEqual(trunc_rv.a, 3)
        self.assertEqual(trunc_rv.b, 7)

    def test_truncated_sampling(self):
        d = UniformDistr(0, 10)
        truncated = d.trunc(2, 8)
        samples = truncated.rand(10000)
        self.assertTrue(np.all(samples >= 2))
        self.assertTrue(np.all(samples <= 8))

    def test_truncated_pdf(self):
        d = NormalDistr(0, 10)
        truncated = d.trunc(2, 8)
        xs_inside = np.linspace(2, 8, 100)
        xs_outside = np.array([1.0, 1.9, 8.1, 10.0])
        Z = d.cdf(8) - d.cdf(2)
        pdf_original = d.pdf(xs_inside)
        pdf_truncated = truncated.pdf(xs_inside)
        self.assertTrue(np.allclose(pdf_original, pdf_truncated * Z, rtol=1e-5))
        pdf_outside = truncated.pdf(xs_outside)
        self.assertTrue(np.allclose(pdf_outside, 0))
