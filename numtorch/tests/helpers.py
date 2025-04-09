from numtorch.layers import LinearLayer, ReluActivation
from numtorch.models import BaseModel, create_sequential_model


def create_perceptron_model(
    in_feature: int, hidden_units: int, out_feature: int
) -> BaseModel:
    layers = [
        LinearLayer(in_feature, hidden_units),
        ReluActivation(),
        LinearLayer(hidden_units, out_feature),
    ]
    return create_sequential_model(layers)
