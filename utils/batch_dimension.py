import numpy as np


def batch_dimension(num_element_dims=2, batch_axis=-1):
    """Decorator which wraps a function so that it can take either a single array, or a batch of arrays.
    The wrapped function should accept and return a batch of arrays.
    If the wrapper detects that a single array has been passed, it reshapes it into a one-element batch.
    """

    def decorator(function):
        """Returns a wrapped version of the function which handles a single array instead of a batch.

        args:
            function: takes in and returns a batch of arrays. Can also return a tuple of results, all with the same bacth axis (e.g. axis=-1)

        returns:
            a version of the function which can take in a single array or a batch.
        """

        def wrapper(x, *args, **kwargs):
            is_single_element = len(x.shape) == num_element_dims
            if is_single_element:
                # Reshape to a batch with 1 element
                x = np.expand_dims(x, axis=batch_axis)

            assert (
                len(x.shape) == num_element_dims + 1
            ), f"expected {num_element_dims}- or {num_element_dims+1}-dimensional array, but got shape {x.shape}"

            result = function(x, *args, **kwargs)

            if is_single_element:
                if isinstance(result, tuple):
                    return tuple(np.squeeze(r, axis=batch_axis) for r in result)
                else:
                    return np.squeeze(result, axis=batch_axis)
            return result

        return wrapper

    return decorator
