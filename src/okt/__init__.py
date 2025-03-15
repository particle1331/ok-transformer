import inspect
from IPython.display import Code


def show(object):
    return lambda: Code(inspect.getsource(object), language="python")
