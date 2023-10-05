import copy
import inspect
import ivy
import importlib
import abc
from typing import List


def load_class_from_str(full_str):
    mod_str = '.'.join(full_str.split('.')[:-1])
    class_str = full_str.split('.')[-1]
    return getattr(importlib.import_module(mod_str), class_str)


def locals_to_kwargs(locals_in):
    keys_to_del = list()
    for k, v in locals_in.items():
        if inspect.isclass(v) or k[0:2] == '__' or k in\
                ['self', 'dataset_dirs', 'dataset_spec', 'data_loader_spec', 'data_loader', 'network_spec', 'network',
                 'trainer_spec', 'trainer', 'tuner_spec', 'tuner']:
            keys_to_del.append(k)
    for key_to_del in keys_to_del:
        del locals_in[key_to_del]
    if 'kwargs' in locals_in:
        kwargs_dict = locals_in['kwargs']
        del locals_in['kwargs']
    else:
        kwargs_dict = {}
    return copy.copy({**locals_in, **kwargs_dict})


class Spec(ivy.Container):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing general properties of the dataset which is saved on disk
        """
        super().__init__(**kwargs)

    @property
    @abc.abstractmethod
    def kwargs(self):
        return self._kwargs


class DatasetDirs(Spec, abc.ABC):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing directories necessary for the data loader
        """
        kw = locals_to_kwargs(locals())
        super().__init__(**kwargs)
        self._kwargs = kw


class DatasetSpec(Spec, abc.ABC):

    def __init__(self,
                 dirs: DatasetDirs,
                 **kwargs) -> None:
        """
        base class for storing general properties of the dataset which is saved on disk
        """
        kw = locals_to_kwargs(locals())
        super().__init__(dirs=dirs,
                         **kwargs)
        self._kwargs = kw


class BaseNetworkSpec(Spec, abc.ABC):

    def __init__(self, dataset_spec: DatasetSpec = None, dev_strs: List[str] = None,
                 v_keychains=None, keep_v_keychains=False, build_mode='explicit', **kwargs) -> None:
        """
        base class for storing general specifications of the neural network
        """
        kw = locals_to_kwargs(locals())
        super().__init__(dataset_spec=dataset_spec,
                         dev_strs=dev_strs,
                         v_keychains=v_keychains,
                         keep_v_keychains=keep_v_keychains,
                         build_mode=build_mode,
                         **kwargs)
        if 'subnets' in self:
            for k, subet_spec in self.subnets.items():
                if 'network_spec_class' in subet_spec:
                    if isinstance(subet_spec.network_spec_class, str):
                        spec_class = load_class_from_str(subet_spec.network_spec_class)
                    else:
                        spec_class = subet_spec.network_spec_class
                    if isinstance(kwargs['subnets'][k], spec_class):
                        subet_spec = kwargs['subnets'][k]
                    else:
                        subet_spec = spec_class(**{**kwargs['subnets'][k],
                                                   **dict(dataset_spec=dataset_spec, dev_strs=dev_strs)})
                    self.subnets[k] = subet_spec
                if isinstance(subet_spec.network_class, str):
                    self.subnets[k].network_class = load_class_from_str(subet_spec.network_class)
                else:
                    self.subnets[k].network_class = subet_spec.network_class
                self.subnets[k].store_vars = ivy.default(self.subnets[k].if_exists('store_vars'), True)
                self.subnets[k].build_mode = ivy.default(self.subnets[k].if_exists('build_mode'), self.build_mode)
                self.subnets[k].dataset_spec = dataset_spec
                self.subnets[k].dev_strs = dev_strs
        self._kwargs = kw


class NetworkSpec(BaseNetworkSpec):

    def __init__(self, dataset_spec, device=None, build_mode='explicit', hypernet=None, network_class=None,
                 **kwargs):

        kw = locals_to_kwargs(locals())

        self.with_hypernetwork = ivy.exists(hypernet)
        if self.with_hypernetwork:
            new_kwargs = dict()
            hypernet_kwargs = hypernet
            hyponet_kwargs = kwargs
            hyponet_kwargs['with_hypernetwork'] = True
            hyponet_kwargs['store_vars'] = False
            new_kwargs['subnets'] = {'hypernet': hypernet_kwargs,
                                     'hyponet': hyponet_kwargs}
            network_class = 'networks.hyper.hyperhyponet.HyperHypoNet'
        else:
            new_kwargs = kwargs
            network_class = ivy.default(network_class, kwargs['network_class'] if 'network_class' in kwargs else None)

        super(NetworkSpec, self).__init__(dataset_spec=dataset_spec,
                                          device=ivy.default_device(device),
                                          build_mode=build_mode,
                                          network_class=network_class,
                                          pass_hypernet_vars=True,
                                          **new_kwargs)

        self._kwargs = kw

    @property
    def kwargs(self):
        return self._kwargs


class BasePerceiverIOSpec(ivy.Container):

    def __init__(self,

                 # input-output dependent
                 input_dim,
                 num_input_axes,
                 output_dim,

                 # input-output agnostic
                 queries_dim=1024,
                 network_depth=8,
                 num_latents=512,
                 latent_dim=1024,
                 num_cross_att_heads=1,
                 num_self_att_heads=8,
                 cross_head_dim=261,
                 latent_head_dim=128,
                 weight_tie_layers=True,
                 learn_query=True,
                 query_shape=None,
                 attn_dropout=0.,
                 fc_dropout=0.,
                 num_lat_att_per_layer=6,
                 cross_attend_in_every_layer=False,
                 with_decoder=True,
                 with_final_head=True,
                 fourier_encode_input=True,
                 num_fourier_freq_bands=6,
                 max_fourier_freq=None,
                 device=None
                 ):

        if learn_query and not ivy.exists(query_shape):
            raise Exception('if learn_query is set, then query_shape must be specified.')

        device = ivy.default(device, ivy.default_device())

        super().__init__(input_dim=input_dim,
                         num_input_axes=num_input_axes,
                         output_dim=output_dim,
                         queries_dim=queries_dim,
                         network_depth=network_depth,
                         num_latents=num_latents,
                         latent_dim=latent_dim,
                         num_cross_att_heads=num_cross_att_heads,
                         num_self_att_heads=num_self_att_heads,
                         cross_head_dim=cross_head_dim,
                         latent_head_dim=latent_head_dim,
                         weight_tie_layers=weight_tie_layers,
                         learn_query=learn_query,
                         query_shape=query_shape,
                         attn_dropout=attn_dropout,
                         fc_dropout=fc_dropout,
                         num_lat_att_per_layer=num_lat_att_per_layer,
                         cross_attend_in_every_layer=cross_attend_in_every_layer,
                         with_decoder=with_decoder,
                         with_final_head=with_final_head,
                         fourier_encode_input=fourier_encode_input,
                         num_fourier_freq_bands=num_fourier_freq_bands,
                         max_fourier_freq=max_fourier_freq,
                         device=device)


class PerceiverIOSpec(NetworkSpec):

    def __init__(self, dataset_spec=None, query_shape=None, **kwargs):

        # kwargs
        kw = locals_to_kwargs(locals())

        # specific kwargs
        kw_p, kw_b = ivy.match_kwargs(kwargs, BasePerceiverIOSpec, NetworkSpec)

        query_shape = ivy.default(query_shape, [])

        # create specification
        super().__init__(
            **BasePerceiverIOSpec(
                query_shape=query_shape, **kw_p),
            dataset_spec=dataset_spec, **kw_b)

        self._kwargs = kw

    @property
    def kwargs(self):
        return self._kwargs