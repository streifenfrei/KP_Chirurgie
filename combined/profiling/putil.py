from torch.autograd import profiler


def profiling(function, label):
    def wrap(*args, **kwargs):
        with profiler.record_function(label):
            return function(*args, **kwargs)

    return wrap
