from .ff_mlp import FF_MLP
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor # Expose block/predictor if needed elsewhere
from .mf_mlp import MF_MLP

# Optional: Define a factory function or dictionary to get models by name
# from torch import nn
# def get_architecture(name: str, **kwargs) -> nn.Module:
#     name = name.lower()
#     if name == 'ff_mlp':
#         return FF_MLP(**kwargs)
#     elif name == 'cafo_cnn':
#         return CaFo_CNN(**kwargs)
#     elif name == 'mf_mlp':
#         return MF_MLP(**kwargs)
#     else:
#         raise ValueError(f"Unknown architecture name: {name}")

__all__ = [
    "FF_MLP",
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    # "get_architecture" # Uncomment if factory is added
]
