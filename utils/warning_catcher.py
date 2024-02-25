import warnings


def catch_warnings(log_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with open(log_path, "w") as f:
                warnings.filterwarnings("always", category=UserWarning)
                warnings.showwarning = lambda *args,: f.write(
                    warnings.formatwarning(*args[:5])
                )
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator
