# model/factory.py
from model.mlp import MLP, mlp_penultimate_getter
from model.moe_d_w import MOE_dense_weight
from .moe_d import MOE_dense, moe_penultimate_getter
from .moe_d_w import moew_penultimate_getter

# from .burgers import BurgersEquation 

def get_model(args):
    """
    根据命令行参数 --eq 自动选择对应的方程类
    """
    mapping = {
        "moe_d": MOE_dense,
        "mlp": MLP,
        "moe_d_w": MOE_dense_weight,
    }
    target_class = mapping.get(args.model.lower())
    if not target_class:
        raise ValueError(f"Model {args.model} is not defined in factory.")
    return target_class(args)

def get_feature_getter(args):
    """
    根据命令行参数 --model 自动选择对应的 feature getter 函数
    """
    mapping = {
        "moe_d": moe_penultimate_getter,
        "mlp": mlp_penultimate_getter,
        "moe_d_w": moew_penultimate_getter,
    }
    target_getter = mapping.get(args.model.lower())
    if not target_getter:
        raise ValueError(f"Feature getter for model {args.model} is not defined in factory.")
    return target_getter