'''Utility functions for smapp_text_classifier
'''
import time
import logging

def timeit(func):
    '''Decorator to print out execution time of funtion'''
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        run_time = time.time() - start
        logging.debug(f'{func.__name__} took {run_time:.2f}s')
        return result
    return timed

def verbose(func):
    '''Decorator that makes a function report everytime it is used'''
    def verbosified(*args, **kwargs):
        logging.debug(f'Using {func.__name__}')
        result = func(*args, **kwargs)
        return result
    return verbosified
