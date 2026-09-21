from stats import mean, moving_average

def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5

def test_moving_average():
    assert moving_average([1, 2, 3, 4, 5], 2) == [1.5, 2.5, 3.5, 4.5]
    assert moving_average([5, 5, 5], 3) == [5.0]
