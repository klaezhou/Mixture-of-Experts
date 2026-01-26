#\Equation/factory.py
from .poisson import PoissonEquation
# from .burgers import BurgersEquation 

def get_equation(args):
    """
    根据命令行参数 --eq 自动选择对应的方程类
    """
    mapping = {
        "poisson": PoissonEquation,
        # "burgers": BurgersEquation,
    }
    target_class = mapping.get(args.eq.lower())
    if not target_class:
        raise ValueError(f"Equation {args.eq} is not defined in factory.")
    return target_class(args)