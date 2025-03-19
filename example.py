from sklearn.datasets import make_moons

from numtorch.layers import BaseLayer, LinearLayer, SigmoidActivation, TanhActivation
from numtorch.models import BaseModel

N_SAMPLES = 200
NOISE = 0.15

x, y = make_moons(N_SAMPLES, noise=NOISE)
x.shape, y.shape


class MyModel(BaseModel):
    def build(
        self, in_feature: int, hidden_units: int, out_feature: int
    ) -> list[BaseLayer]:
        return [
            LinearLayer(in_feature, hidden_units),
            TanhActivation(),
            LinearLayer(hidden_units, out_feature),
            TanhActivation(),
            SigmoidActivation(),
        ]
