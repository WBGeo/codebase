# Pydantic adapter for panda DataFrame and Meshio CellBlocks
import codecs
import pickle
import typing

import meshio
import pandas as pd
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class _PandasDFPydanticAnnotation:
  from pydantic import GetJsonSchemaHandler
  import typing
  from typing import Any, Callable
  @classmethod
  def __get_pydantic_core_schema__(
      cls,
      _source_type: typing.Any,
      _handler: Callable[[Any], core_schema.CoreSchema],
  ) -> core_schema.CoreSchema:

    def validate(value: typing.Any) -> pd.DataFrame:
      if isinstance(value, pd.DataFrame):
        return value
      elif isinstance(value, list):
        return pd.DataFrame(value)
      raise TypeError("Expected a pandas DataFrame or a list of dictionaries.")

    my_schema = core_schema.chain_schema(
      [
        core_schema.union_schema([
          core_schema.is_instance_schema(pd.DataFrame),
          core_schema.dict_schema(),
          core_schema.list_schema(),
        ]),
        core_schema.no_info_plain_validator_function(validate),
      ]
    )

    return core_schema.json_or_python_schema(
      json_schema=my_schema,
      python_schema=my_schema,
      serialization=core_schema.plain_serializer_function_ser_schema(
        lambda df: df.to_dict(orient="records")
      ),
    )

  @classmethod
  def __get_pydantic_json_schema__(
      cls, field_core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
  ) -> JsonSchemaValue:
    return handler(core_schema.union_schema([
      core_schema.is_instance_schema(pd.DataFrame),
      core_schema.dict_schema()
    ]))


PandasDataFrame = typing.Annotated[pd.DataFrame, _PandasDFPydanticAnnotation]


class _MeshIOCellBlockPydanticAnnotation:
  from pydantic import GetJsonSchemaHandler
  import typing
  from typing import Any, Callable
  @classmethod
  def __get_pydantic_core_schema__(
      cls,
      _source_type: typing.Any,
      _handler: Callable[[Any], core_schema.CoreSchema],
  ) -> core_schema.CoreSchema:

    def validate(value: typing.Any) -> meshio.CellBlock:
      if isinstance(value, meshio.CellBlock):
        return value
      elif value is None:
        return None
      elif isinstance(value, str):
        return pickle.loads(codecs.decode(value.encode(), "base64"))
      raise TypeError("Expected a meshio.CellBlock or (base64 encoded) string")

    from_str_schema = core_schema.chain_schema(
      [
        core_schema.str_schema(),
        core_schema.no_info_plain_validator_function(validate),
      ]
    )

    return core_schema.json_or_python_schema(
      json_schema=from_str_schema,
      python_schema=core_schema.union_schema([
        core_schema.is_instance_schema(meshio.CellBlock),
        from_str_schema,
      ]),
      serialization=core_schema.plain_serializer_function_ser_schema(
        lambda obj: codecs.encode(pickle.dumps(obj), "base64").decode()
      ),
    )

  @classmethod
  def __get_pydantic_json_schema__(
      cls, field_core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
  ) -> JsonSchemaValue:
    return handler(core_schema.union_schema([
      core_schema.str_schema()
    ]))


MeshIOCellBlock = typing.Annotated[meshio.CellBlock, _MeshIOCellBlockPydanticAnnotation]
