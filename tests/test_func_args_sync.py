from inspect import isfunction, signature
from typing import Any, cast, get_args

from langchain_core.tools import ArgsSchema
from pydantic import SecretStr

from dean_research_tools import PGTools
from dean_research_tools.config import Settings
from dean_research_tools.retriever import AVAILABLE_TOOLS


def _safe_schema(model: ArgsSchema) -> dict[str, Any]:
    if isinstance(model, dict):
        return model
    attr_names = ["model_json_schema", "schema"]
    for attr_name in attr_names:
        if hasattr(model, attr_name):
            return getattr(model, attr_name)()

    raise ValueError(
        f"Model {model} does not have a schema method or model_json_schema method"
    )


TYPE_MAP = {
    "string": "str",
    "null": "None",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _parse_model_arg(model_args: dict[str, Any], arg: str) -> set[str]:
    if arg not in model_args:
        raise ValueError(f"Argument {arg} not found in model args")
    arg_info = model_args[arg]
    return _parse_arg_info(arg_info)


def _parse_arg_info(arg_info: dict[str, Any]) -> set[str]:
    inner_types = ""
    if "items" in arg_info:
        inner_types_raw = sorted(list(_parse_arg_info(arg_info["items"])))
        inner_types = "[" + ",".join(inner_types_raw) + "]"
    if "type" in arg_info:
        arg_type = TYPE_MAP[arg_info["type"]]
        return {arg_type + inner_types}
    if "anyOf" in arg_info:
        types = set()
        for option in arg_info["anyOf"]:
            types.update(_parse_arg_info(option))
        return types
    raise ValueError(f"Argument info {arg_info} does not have a type or anyOf field")


def _alphabatise_inner(annotation: str) -> str:
    open_indx = annotation.find("[")
    while open_indx > 0:
        close_indx = annotation.find("]", open_indx)
        inner = annotation[open_indx + 1 : close_indx]
        inner_parts = [p.strip() for p in inner.split(",")]
        inner_parts.sort()
        inner_sorted = ",".join(inner_parts)
        annotation = (
            annotation[: open_indx + 1] + inner_sorted + annotation[close_indx:]
        )
        open_indx = annotation.find("[", close_indx)
    return annotation


def _parse_func_arg(annotation: str) -> set[str]:
    annotation = _alphabatise_inner(annotation)
    split_chars = [",", "|"]
    for char in split_chars:
        if char in annotation:
            parts = [p.strip() for p in annotation.split(char)]
            return {p for p in parts}

    return {annotation}


blank = Settings(
    db_host="a",
    db_port=1,
    db_name="b",
    db_user="c",
    db_password="d",
    azure_openai_api_key=SecretStr("e"),
    azure_embedding_endpoint="f",
    azure_openai_embedding_deployment="g",
)


def test_get_all_tools():
    """Tests that the functions returned by _get_all_tools match the functions defined in the PGTools class."""
    funcs_in_all = set(x.name for x in PGTools(settings=blank)._get_all_tools())

    funcs_in_class = set(
        x for x in dir(PGTools) if x[0] != "_" and isfunction(getattr(PGTools, x))
    )

    if funcs_in_class != funcs_in_all:
        missing_from_class = funcs_in_all - funcs_in_class
        missing_from_function = funcs_in_class - funcs_in_all
        raise ValueError(
            f"Mismatch between PGTools class and _get_all_tools. Missing from class: {missing_from_class}. Missing from function: {missing_from_function}"
        )


def test_available_tools():
    """Tests that the functions defined in the PGTools class match the functions listed in AVAILABLE_TOOLS."""
    funcs_in_class = set(
        x for x in dir(PGTools) if x[0] != "_" and isfunction(getattr(PGTools, x))
    )

    funcs_in_literal = set(get_args(AVAILABLE_TOOLS))

    if funcs_in_class != funcs_in_literal:
        missing_from_class = funcs_in_literal - funcs_in_class
        missing_from_literal = funcs_in_class - funcs_in_literal
        raise ValueError(
            f"Mismatch between PGTools class and AVAILABLE_TOOLS literal. Missing from class: {missing_from_class}. Missing from literal: {missing_from_literal}"
        )


def test_arg_schema_sync():
    """Tests that the function arguments defined in the PGTools class match the arguments defined in the ArgsSchema for each tool."""
    for struct_tool in PGTools(settings=blank)._get_all_tools():
        func_args = signature(getattr(PGTools, struct_tool.name)).parameters
        model_args = _safe_schema(struct_tool.args_schema)["properties"]
        for arg in func_args:
            if arg == "self":
                continue
            func_annot = str(func_args[arg].annotation)

            model_types = _parse_model_arg(model_args, arg)
            func_types = _parse_func_arg(func_annot)
            assert func_types == model_types
