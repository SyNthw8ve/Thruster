import itertools

def build_tests(param_grid):

        test_keys = param_grid.keys()

        test_values = param_grid.values()
        test_combinations = itertools.product(*test_values)

        test_items = [dict(zip(test_keys, test_item)) for test_item in test_combinations]

        return test_items