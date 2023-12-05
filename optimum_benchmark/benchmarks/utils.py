from typing import Callable


def extract_three_significant_digits(x: float) -> float:
    return float(f"{x:.3g}")


def three_significant_digits_wrapper(func: Callable[..., float]) -> Callable[..., float]:
    def wrapper(*args, **kwargs):
        return extract_three_significant_digits(func(*args, **kwargs))

    return wrapper
