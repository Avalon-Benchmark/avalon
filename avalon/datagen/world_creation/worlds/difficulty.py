from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import numpy as np


def get_difficulty_based_value(
    difficulty: float, min_val: float, max_val: float, variability: float, rand: np.random.Generator
) -> float:
    total_delta = max_val - min_val
    delta = variability * total_delta
    remainder = total_delta - delta
    return min_val + (remainder * difficulty) + (rand.uniform() * delta)


T = TypeVar("T")


def select_categorical_difficulty(
    choices: Sequence[T],
    difficulty: float,
    rand: np.random.Generator,
    _FORCED: Optional[T] = None,
) -> Tuple[T, float]:
    """
    returns selected choice, new difficulty
    """

    num_choices = len(choices)
    prob_coeff = 1.0 / sum(difficulty ** float(x) for x in range(num_choices))
    choice_prob = [
        difficulty / num_choices + (1 - difficulty) * (difficulty**x) * prob_coeff for x in range(num_choices)
    ]
    choice_idx = rand.choice(range(num_choices), p=choice_prob)
    if _FORCED is not None:
        assert _FORCED in choices
        choice_idx = choices.index(_FORCED)

    # TODO use other less arbitrary method to calculate?
    new_difficulty = difficulty**2 + (1 - difficulty) * (difficulty ** (choice_idx + 1))
    return choices[choice_idx], new_difficulty


def select_boolean_difficulty(
    difficulty: float,
    rand: np.random.Generator,
    initial_prob: float = 1,
    final_prob: float = 0.01,
    _FORCED: Optional[bool] = None,
) -> Tuple[bool, float]:
    """
    Uses log interpolation to scale initial prob (difficulty=0) to final prob (difficulty=1). Zero probability is not
    valid, but 1 is. Rescales difficulty to account for probability distribution.
    Tips:
    - Always finish boolean choices before using scalars
    - Somewhat better to have easier option be "true", since we can set p(True) = 1 at difficulty 0.

    debug_override_value that is not None will ALWAYS return that value!
    """
    sampled_value = rand.uniform()
    if _FORCED is not None:
        sampled_value = 0.0 if _FORCED else 1.0

    assert initial_prob <= 1 and final_prob <= 1, "Probability cannot be greater than 1!"
    if initial_prob == final_prob:
        return sampled_value < initial_prob, difficulty
    assert initial_prob > 0 and final_prob > 0, "Cannot have zero probability for True with log interpolation!"
    prob = (initial_prob ** (1 - difficulty)) * (final_prob**difficulty)
    value = sampled_value < prob
    # the updated difficulty is the integral of the probability to d over the integral to 1
    if value:
        new_difficulty = (prob - initial_prob) / (final_prob - initial_prob)
    else:
        prob_coeff = np.log(initial_prob) - np.log(final_prob)
        new_difficulty = (prob - initial_prob + difficulty * prob_coeff) / (final_prob - initial_prob + prob_coeff)
    # the abs here ensures that we don't return -0.0, which caused an annoying bug before
    return value, np.clip(new_difficulty, 0.0, 1.0)


def scale_with_difficulty(
    difficulty: float, start_val: float, end_val: float, _FORCED: Optional[float] = None
) -> float:
    if _FORCED:
        return _FORCED
    delta = end_val - start_val
    return start_val + delta * difficulty


def difficulty_variation(
    start_val: float,
    end_val: float,
    rand: np.random.Generator,
    difficulty: float,
    _FORCED: Optional[float] = None,
) -> float:
    if _FORCED:
        return _FORCED
    delta = end_val - start_val
    return start_val + delta * difficulty * rand.uniform()


def normal_distrib_range(
    start_val: float,
    end_val: float,
    std_dev: float,
    rand: np.random.Generator,
    difficulty: float,
    _FORCED: Optional[float] = None,
) -> float:
    if _FORCED:
        return _FORCED
    delta = end_val - start_val
    mean = start_val + delta * difficulty
    min_val = min([start_val, end_val])
    max_val = max([start_val, end_val])
    return float(np.clip(rand.normal(mean, std_dev), min_val, max_val))


def get_rock_probability(difficulty: float) -> float:
    return scale_with_difficulty(difficulty, 0.5, 0.95)
