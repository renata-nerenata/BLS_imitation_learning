from data.utils import get_tenzor_encoding_input, get_tenzor_encoding_output
from metrics.metrics import get_cosine_similarity
import pytest
import numpy as np
from tests.fixtures import *


def test_get_tenzor_encoding_input():
    assert get_tenzor_encoding_input(puzzle=PUZZLE, elements=STANDART) == ENCODE_PUZZLE


def test_get_tenzor_encoding_output():
    assert get_tenzor_encoding_output(puzzle=SOLUTION) == ENCODE_SOLUTION


def test_get_cosine_similarity():
    a = np.array([[1, 0, -1]])
    b = np.array([[-1, -1, 0]])
    assert round(get_cosine_similarity(a, b), 1) == -0.5
