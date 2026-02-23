import inspect
import sys
import types
import typing
import unittest
from typing import Unpack, Any, Optional

from py_api_wbgeo import nodesapi
from py_api_wbgeo.apitypes import ComponentDecoratorParams, RegisterScriptBlockParams, \
    APIScriptBlockDefinition, \
    RegisterVisualizerParams, GeoExecuteAPI, ScriptTypeParams
from py_api_wbgeo.nodesapi import AnnotatedScriptType


class TestComponentSignatures(unittest.TestCase):
    """
    Test if all components (within the core directory) are properly defined using decorators
    i.e.: all parameters are annotated, functions properly decorated, etc.

    Also tests if all imports are correct
    """

    @classmethod
    def setup_class(cls):
        from pathlib import Path

        core = Path("core")
        if not core.exists():
            core = Path("../core")
            if core.exists():
                # workaround for our friends using the IDE instead of a CLI :)
                print("NOTICE: Test started not from root directory - adding root to path:",
                      core.parent.resolve())
                sys.path.append(str(core.parent.resolve()))
            else:
                raise ValueError("Unable to find core directory")

        print("Using core folder found in", core)
        components_folder = core.resolve()
        codebase_folder = core.parent.resolve()
        cls.py_files = [".".join(py_file.relative_to(codebase_folder).parts)[:-3] for py_file in
                        components_folder.rglob("*.py") if py_file.stem != "__init__"]
        nodesapi.set_instance(MockBackendInstance())
        # force unload all py-files to allow decorators to actually work
        for py_file in cls.py_files:
            if py_file in sys.modules:
                del sys.modules[py_file]
        print("with", cls.py_files)

    @classmethod
    def teardown_class(cls):
        # and unload/reset the decorated modules
        nodesapi.set_instance(None)
        for py_file in cls.py_files:
            if py_file in sys.modules:
                del sys.modules[py_file]

    def test_load_components(self):
        import importlib
        for py_file in self.py_files:
            importlib.import_module(py_file)


def get_location(x):
    return inspect.getfile(x) + ":" + str(inspect.getsourcelines(x)[1])


class MockBackendInstance:
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
            import typing
            return typing.Annotated[c, AnnotatedScriptType(**kwargs)]

        return h

    def wbgeo_inspector_decorator(self, **kwargs: Unpack[nodesapi.InspectorParams]):
        def do_work(f):
            origin = typing.get_origin(f)
            if origin is typing.Annotated:
                args = typing.get_args(f)
                func, *metadata = args
                ast = next((m for m in metadata if isinstance(m, ComponentMethod)), None)
                if ast:
                    raise Exception(
                        "The @wbgeo_inspector must be added as the line AFTER @wbgeo_component",
                        get_location(f))
            return typing.Annotated[f, VisualizerComponent()]

        return lambda f_component: do_work(f_component)

    def wbgeo_component_decorator(self, **kwargs: Unpack[ComponentDecoratorParams]):
        params: ComponentDecoratorParams = kwargs

        def gwt_ast(__metadata__: tuple) -> Optional[AnnotatedScriptType]:
            for v in __metadata__:
                if isinstance(v, AnnotatedScriptType):
                    return v
            return None

        def decorate_func(f):
            from inspect import signature, Signature

            origin = typing.get_origin(f)
            is_visualizer = False
            if origin is typing.Annotated:
                args = typing.get_args(f)
                func, *metadata = args
                vis_comp = next((m for m in metadata if isinstance(m, VisualizerComponent)), None)
                if vis_comp:
                    is_visualizer = True
                    f = args[0]
            sig = signature(f)
            if sig.return_annotation == Signature.empty:
                if not is_visualizer:
                    raise ValueError(
                        "@GeoComponent requires a declared return type, e.g. @GeoComponent def {fname}() -> str: \n{floc}".format(
                            fname=str(f.__name__), floc=get_location(f)))
            elif sig.return_annotation in [str, bool, int, float]:
                pass
            elif not hasattr(sig.return_annotation, '__metadata__'):
                raise ValueError(
                    "GeoComponent must return a built-in type or a type annotated with AnnotatedScriptType, instead it returned `{t}`, {typel} / {floc}".format(
                        t=str(sig.return_annotation), typel=get_location(sig.return_annotation),
                        floc=get_location(f)))
            else:
                ret_v = gwt_ast(sig.return_annotation.__metadata__)

                if ret_v is None:
                    raise ValueError(
                        "Unsupported return type " + sig.return_annotation + str(sig) + get_location(f))
                else:
                    pass

            ### end return
            def get_type_from_param(t, param: str, loc, is_vis: bool):
                # t = sig.parameters[param].annotation
                origin = typing.get_origin(t)
                if t is None or t == inspect.Parameter.empty:
                    raise ValueError(
                        "Parameter `{param}` must be annotated, e.g. `{param}: str` or `{param}: str = 1` \n {loc}".format(
                            param=param, loc=get_location(t)))
                if not hasattr(t, '__metadata__') or gwt_ast(t.__metadata__) is None:
                    if origin is typing.Union:
                        u_args = typing.get_args(t)
                        if len(u_args) == 2 and u_args[1] == types.NoneType:
                            return get_type_from_param(u_args[0], param, loc, is_vis)
                    elif t in [str, bool, int, float]:
                        pass
                    elif is_vis and str(t) == '<class \'py_api_wbgeo.nodesapi.InspectorHelper\'>':
                        pass
                    else:
                        raise ValueError(
                            "@GeoComponent parameter `{param}` must be a built-in type or its type `{type}` be annotated with AnnotatedScriptType at {loc}".format(
                                param=param, type=str(t), loc=get_location(loc)))
                else:
                    pass

            for param in sig.parameters:
                t = sig.parameters[param].annotation
                get_type_from_param(t, param, f, is_visualizer)
            return f

        return decorate_func


class ComponentMethod():
    def __init__(self, identifier: str):
        self.identifier = identifier


class VisualizerComponent:
    pass


if __name__ == '__main__':
    unittest.main()
