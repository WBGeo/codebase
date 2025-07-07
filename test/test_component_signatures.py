import inspect
import sys
import types
import typing
import unittest
from pathlib import Path
from typing import Unpack, Any, Optional

import os
from py_api_wbgeo import nodesapi
from py_api_wbgeo.apitypes import ComponentDecoratorParams, RegisterScriptBlockParams, APIScriptBlockDefinition, \
    RegisterVisualizerParams, GeoExecuteAPI, ScriptTypeParams
from py_api_wbgeo.nodesapi import AnnotatedScriptType


# Test if all components are properly defined using decorators
# i.e.: all parameters are annotated
class TestInputData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        class MockBackendInstance:
            def register_script_block(self, **kwargs: Unpack[RegisterScriptBlockParams]) -> APIScriptBlockDefinition:
                pass

            def register_visualizer(self, **kwargs: Unpack[RegisterVisualizerParams]):
                pass

            def create_geo_execute(self, exec: Any) -> GeoExecuteAPI:
                pass

            def wbgeo_type_decorator(self, **kwargs: Unpack[ScriptTypeParams]):
                def h(c):
                    import typing
                    return typing.Annotated[c, AnnotatedScriptType(**kwargs)]

                return h

            def wbgeo_component_decorator(self, **kwargs: Unpack[ComponentDecoratorParams]):
                params: ComponentDecoratorParams = kwargs

                def gwt_ast(__metadata__: tuple) -> Optional[AnnotatedScriptType]:
                    for v in __metadata__:
                        if isinstance(v, AnnotatedScriptType):
                            return v
                    return None

                def decorate_func(f):
                    from inspect import signature, Signature
                    sig = signature(f)
                    if sig.return_annotation == Signature.empty:
                        raise ValueError(
                            "@GeoComponent requires a declared return type, e.g. @GeoComponent def myDef() -> str: ..." + str(
                                sig))
                    elif sig.return_annotation in [str, bool, int, float]:
                        pass
                    elif not hasattr(sig.return_annotation, '__metadata__'):
                        raise ValueError(
                            "GeoComponent must return a built-in type or a type annotated with AnnotatedScriptType, instead it returned `{t}`".format(
                                t=str(sig.return_annotation)))
                    else:
                        ret_v = gwt_ast(sig.return_annotation.__metadata__)

                        if ret_v is None:
                            raise ValueError("Unsupported return type " + sig.return_annotation + str(sig))
                        else:
                            pass

                    ### end return
                    def get_type_from_param(t, param: str):
                        # t = sig.parameters[param].annotation
                        origin = typing.get_origin(t)
                        if t is None or t == inspect.Parameter.empty:
                            raise ValueError(
                                "Parameter `{param}` must be annotated, e.g. `{param}: str` or `{param}: str = 1` ".format(
                                    param=param))
                        if not hasattr(t, '__metadata__') or gwt_ast(t.__metadata__) is None:
                            if origin is typing.Union:
                                u_args = typing.get_args(t)
                                if len(u_args) == 2 and u_args[1] == types.NoneType:
                                    return get_type_from_param(u_args[0], param)
                            elif t in [str, bool, int, float]:
                                pass
                            else:
                                raise ValueError(
                                    "@GeoComponent parameter `{param}` must be a built-in type or its type `{type}` be annotated with AnnotatedScriptType".format(
                                        param=param, type=str(t)))
                        else:
                            pass

                    for param in sig.parameters:
                        t = sig.parameters[param].annotation
                        get_type_from_param(t, param)
                    return f

                return decorate_func

        nodesapi.set_instance(MockBackendInstance())
        print("Setting instabnce")

    def test_load_components(self):
        from pathlib import Path
        import importlib
        components_folder = Path("core").resolve()
        codebase_folder = Path(".").resolve()
        print(sys.path)
        for py_file in components_folder.rglob("*.py"):
            if py_file.stem == "__init__":
                continue
            module_name = ".".join(py_file.relative_to(codebase_folder).parts)
            importlib.import_module(module_name[:-3])


if __name__ == '__main__':
    sys.path.append(str(Path(os.path.dirname(__file__) + "/..").resolve()))
    unittest.main()
