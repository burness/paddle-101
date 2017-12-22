# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import re
from paddle.trainer_config_helpers.default_decorators import wrap_name_default
import paddle.trainer_config_helpers as conf_helps


class LayerType(type):
    def __new__(cls, name, bases, attrs):
        method_name = attrs.get('METHOD_NAME', None)
        if method_name is not None:
            method = getattr(conf_helps, method_name)
            if method.__doc__ is not None:
                mapper = attrs.get("__map_docstr__", None)
                if mapper is not None:
                    attrs['__doc__'] = LayerType.__map_docstr__(
                        mapper(method.__doc__),
                        method_name=method_name,
                        name=name)
                else:
                    attrs['__doc__'] = LayerType.__map_docstr__(
                        method.__doc__, method_name=method_name, name=name)
        return super(LayerType, cls).__new__(cls, name, bases, attrs)

    @staticmethod
    def __map_docstr__(doc, name, method_name):
        assert isinstance(doc, basestring)

        # replace LayerOutput to paddle.v2.config_base.Layer
        doc = doc.replace("LayerOutput", "paddle.v2.config_base.Layer")

        doc = doc.replace('ParameterAttribute',
                          'paddle.v2.attr.ParameterAttribute')

        doc = re.sub(r'ExtraLayerAttribute[^\s]?',
                     'paddle.v2.attr.ExtraAttribute', doc)

        # xxx_layer to xxx
        doc = re.sub(r"(?P<name>[a-z]+)_layer", r"\g<name>", doc)

        # XxxxActivation to paddle.v2.Activation.Xxxx
        doc = re.sub(r"(?P<name>[A-Z][a-zA-Z]+)Activation",
                     r"paddle.v2.Activation.\g<name>", doc)

        # TODO(yuyang18): Add more rules if needed.
        return doc


class Layer(object):
    __metaclass__ = LayerType

    def __init__(self, name=None, parent_layers=None):
        assert isinstance(parent_layers, dict)
        self.name = name
        self.__contex__ = {}
        self.__parent_layers__ = parent_layers

    def to_proto(self, context):
        """
        function to set proto attribute
        """
        kwargs = dict()
        for layer_name in self.__parent_layers__:
            if not isinstance(self.__parent_layers__[layer_name],
                              collections.Sequence):
                v1_layer = self.__parent_layers__[layer_name].to_proto(
                    context=context)
            else:
                v1_layer = map(lambda x: x.to_proto(context=context),
                               self.__parent_layers__[layer_name])
            kwargs[layer_name] = v1_layer

        if self.context_name() is None:
            return self.to_proto_impl(**kwargs)
        elif self.context_name() not in context:
            context[self.context_name()] = self.to_proto_impl(**kwargs)
        self.__contex__ = context
        if self.use_context_name():
            return context[self.context_name()]
        else:
            return context[self.name]

    def to_proto_impl(self, **kwargs):
        raise NotImplementedError()

    def context_name(self):
        """
        Context name means the context which stores `to_proto_impl` result.
        If multiple layer share same context_name, the `to_proto_impl` of them
        will be invoked only once.
        """
        return self.name

    def use_context_name(self):
        return False

    def calculate_size(self):
        """
        lazy calculate size of the layer, should be called when to_proto_impl of
        this layer is called.
        :return:
        """
        return self.__contex__[self.context_name()].size


def __convert_to_v2__(method_name, parent_names, is_default_name=True):
    if is_default_name:
        wrapper = wrap_name_default(name_prefix=method_name)
    else:
        wrapper = None

    class V2LayerImpl(Layer):
        METHOD_NAME = method_name

        def __init__(self, **kwargs):
            parent_layers = dict()
            other_kwargs = dict()
            for pname in parent_names:
                if kwargs.has_key(pname):
                    parent_layers[pname] = kwargs[pname]

            for key in kwargs.keys():
                if key not in parent_names:
                    other_kwargs[key] = kwargs[key]

            name = kwargs.get('name', None)
            super(V2LayerImpl, self).__init__(name, parent_layers)
            self.__other_kwargs__ = other_kwargs

        if wrapper is not None:
            __init__ = wrapper(__init__)

        def to_proto_impl(self, **kwargs):
            args = dict()
            for each in kwargs:
                args[each] = kwargs[each]
            for each in self.__other_kwargs__:
                args[each] = self.__other_kwargs__[each]
            return getattr(conf_helps, method_name)(**args)

    return V2LayerImpl
