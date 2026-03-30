"""
This utility adds some checks to simulate the runtime interpreter
 - every MyGeoType(...) constructor call is intercepted to perform serialization & deserialization
    - MyGeoType(x=1) ==> MyGeoType(**json.loads(MyGeoType(x=1).model_dump_json))

  Register it by calling register_as_test_nodes_api()
"""
import typing


def register_as_test_nodes_api():
  import json
  from typing import Unpack, Any
  import io
  import dataclasses
  import pydantic_core
  import sys
  import pydantic
  from py_api_wbgeo import nodesapi
  from py_api_wbgeo.apitypes import APIScriptBlockDefinition, RegisterVisualizerParams, \
    GeoExecuteAPI, ScriptTypeParams, InspectorParams, ComponentDecoratorParams, \
    RegisterScriptBlockParams

  def dump_as_json(ret: typing.Any):
    if type(ret) == str or type(ret) == int or type(ret) == bool:
      return str(ret)
    elif isinstance(ret, pydantic.BaseModel):
      return ret.model_dump_json(exclude_none=True)  # todo exclude_none=True always?
    elif isinstance(ret, io.BytesIO):
      raise Exception("BytesIO not supported in mocked test")
    elif isinstance(ret, nodesapi.GeoTempFile):
      raise Exception("GeoTempFile not supported in mocked test")
    else:
      from pydantic_core import to_jsonable_python
      return json.dumps(to_jsonable_python(ret))

  class PydanticExampleBackendInstance:
    """
    backend instance
    """

    def register_script_block(self, **kwargs: Unpack[
      RegisterScriptBlockParams]) -> APIScriptBlockDefinition:
      raise ValueError("not supported!")

    def register_visualizer(self, **kwargs: Unpack[RegisterVisualizerParams]):
      raise ValueError("not supported!")

    def create_geo_execute(self, exec: Any) -> GeoExecuteAPI:
      raise ValueError("not supported!")

    def wbgeo_type_decorator(self, **kwargs: Unpack[ScriptTypeParams]):
      params: ScriptTypeParams = kwargs

      def h(c):
        print("Wrapping around" + str(c) + "/" + str(type(c)))

        def constructor_fn(model_cls, *args, **kwargs):
          orig = model_cls(*args, **kwargs)
          return copy_fn(model_cls, orig)

        def copy_fn(model_cls, orig):
          # serialize
          try:
            as_json = dump_as_json(orig)
          except Exception as e:
            e.args = f"Failed to serialize {str(model_cls)} to json. \nCheck the original message for more details:\n{e.args}",
            raise
          # deserialize and construct real new instance
          import json
          try:
            copy_after_de_serialization = model_cls(**json.loads(as_json))
          except Exception as e:
            e.args = f"Failed to deserialize json to {str(model_cls)}.\nCheck the original message for more details:\n{e.args}",
            raise
          # then return new instance: Parameters might be missing :)
          return copy_after_de_serialization

        class _WrapperMeta(type(c)):
          def __call__(cls, *args, **kwargs):
            # delegate construction
            return constructor_fn(c, *args, **kwargs)

          def __instancecheck__(cls, instance):
            return isinstance(instance, c)

        class Wrapped(c, metaclass=_WrapperMeta):
          def __copy__(self):
            raise ValueError("__copy__ not supported. Try colors of my dream instead")

          def __deepcopy__(self, memo):
            return copy_fn(c, self)  # wrap deepcopy to

          def __eq__(self, other):
            # Compare to either the wrapped class or itself (to support ==)
            if isinstance(other, c):
              # pydantic __eq__ does an exact type check -> we have to make it ourself
              if isinstance(c, pydantic.BaseModel):
                return self.model_dump() == other.model_dump()
              elif dataclasses.is_dataclass(c):
                return pydantic_core.to_json(self) == pydantic_core.to_json(other)
              return c.__eq__(self, other)
            return NotImplemented

        Wrapped.__name__ = f"{c.__name__}WrappedGeoType"
        return Wrapped

      return lambda c: h(c)

    def wbgeo_inspector_decorator(self, **kwargs: Unpack[InspectorParams]):
      return lambda f_component: f_component

    def wbgeo_component_decorator(self, **kwargs: Unpack[ComponentDecoratorParams]):
      params: ComponentDecoratorParams = kwargs
      return lambda f_component: f_component

  nodesapi.set_instance(PydanticExampleBackendInstance())
  print("Setting nodesapi instance to simulate the runtime", file=sys.stderr)
  print(" In case of errors, please check that you followed proper pydantic guidelines",
        file=sys.stderr)
  print(
    " In case of missing/null data, please check that no required fields are marked as private and (de)serializers are defined",
    file=sys.stderr)
