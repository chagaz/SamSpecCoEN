# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# January 2016


"""utils.py: Utility functions."""

import os
import sys

def wrapper(func, *args, **kwargs):
    ''' Wrap a function with arguments into a function without arguments.
    Can be used in particular to pass it to timeit.

    Parameters
    ----------
    func: function
        Function to wrap.

    *args: function arguments
        Arguments to pass.

    **kwargs: function keyword parameters
        Keyword parameters to pass.

    Returns
    ------
    wrapped: function
        Function with no arguments, equivalent to the one passed with arguments.

    Reference
    ---------
    http://pythoncentral.io/time-a-python-function/
    '''
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


