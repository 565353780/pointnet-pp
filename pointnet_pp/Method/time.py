from time import time


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()
