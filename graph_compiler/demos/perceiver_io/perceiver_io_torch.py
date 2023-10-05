import ivy
import inspect
from perceiver_pytorch import PerceiverIO as PerceiverTorch
from perceiver_io.perceiver_io_ivy import BaseNetwork


class PerceiverIOTorch(BaseNetwork):

    def __init__(self, spec, v=None):
        super(PerceiverIOTorch, self).__init__(spec, v=v)

    def _build(self, *args, **kwargs):
        full_arg_spec = inspect.getfullargspec(PerceiverTorch.__init__)
        expected_kwargs = full_arg_spec.kwonlyargs
        spec_to_pass = dict([(k, v) for k, v in self._spec.items() if k in expected_kwargs])
        spec_to_pass["dim"] = self._spec["input_dim"]
        spec_to_pass["depth"] = self._spec["num_input_axes"]
        self._network = ivy.to_ivy_module(
            PerceiverTorch(**spec_to_pass).to(self._dev))

    def _forward(self, data, mask=None, queries=None):
        # remove time dimension if un-unsed
        if data.shape[1] == 1:
            data = ivy.squeeze(data, 1)
        if data.shape[0] == 1:
            data = ivy.squeeze(data, 0)
        return self._network(data, mask, queries)
