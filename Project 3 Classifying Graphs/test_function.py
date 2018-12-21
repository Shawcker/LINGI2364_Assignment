from main import RemoveX1FromX2
import pytest


def test_RemoveX1FromX2():
    X1 = [[1, 2], [2, 3], []]
    X2= [[1, 2, 3], [2, 3, 4, 5], [1]]
    result = RemoveX1FromX2(X1, X2)
    assert result == [[3], [4, 5], [1]], result