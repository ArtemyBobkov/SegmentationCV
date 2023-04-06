import numpy as np
import torch
from torch.utils.data import Subset

from src.metrics import *
from src.utils import assert_close
import pytest


class TestMetric:

    @pytest.mark.parametrize('pred,label',
                             zip(Subset(pytest.dataset, np.arange(10)),
                                 Subset(pytest.dataset, np.arange(10))))
    def test_DiffFgDICE_true(self, pred, label):
        answer = DiffFgDICE(pred['sem'], label['sem'])
        assert_close(answer, 1.)

    def test_DiffFgDICE_false_full(self):
        answer = DiffFgDICE(torch.zeros((3, 256, 256)), torch.ones((3, 256, 256)))
        assert_close(answer, 0)

    @pytest.mark.parametrize('pred,label',
                             zip(Subset(pytest.dataset, np.arange(10)),
                                 Subset(pytest.dataset, np.arange(10))))
    def test_DiffFgDICE_real(self, pred, label):
        answer = DiffFgDICE(pred['sem'].max() - pred['sem'], label['sem'])
        assert_close(answer, 0)

    @pytest.mark.parametrize('pred,label',
                             zip(Subset(pytest.dataset, np.arange(10)),
                                 Subset(pytest.dataset, np.arange(10))))
    def test_DiffFgMSE_true(self, pred, label):
        answer = DiffFgMSE(pred['sem'], label['sem'])
        assert_close(answer, 0.)

    def test_DiffFgMSE_false_full(self):
        answer = DiffFgMSE(torch.zeros((3, 256, 256)), torch.ones((3, 256, 256)))
        assert answer > 1e2

    @pytest.mark.parametrize('pred,label',
                             zip(Subset(pytest.dataset, np.arange(10)),
                                 Subset(pytest.dataset, np.arange(10))))
    def test_DiffFgMSE_real(self, pred, label):
        answer = DiffFgMSE(pred['sem'].max() - pred['sem'], label['sem'])
        assert answer > 1
