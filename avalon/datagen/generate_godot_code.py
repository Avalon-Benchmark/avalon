"""
Automatically generates code on both the godot side and python side.

For godot, simply creates the deserialization function because godot is too simple
to have a global registry of classes.

For python, creates attrs types for the types that represent useful objects in godot.

The godot side is purposefully restricted to only deserializing Builder types.
Builder types are those which can actually be created from a config.
Spec types are simply the definitions of fields for the Builders.
THis separation is done so that the configured, created objects can have a type-safe
reference to the stateless Spec class.

Currently python distribution classes are not auto generated, but are included in
godot_base_types.py.  This could be fixed in a future version.
"""

import itertools
import re
from pathlib import Path
from typing import Dict
from typing import Final
from typing import Iterable
from typing import List
from typing import Set
from typing import TypeVar

COMMENTS: Final = {
    "SimSpec": "TODO: should SimSpec still inherit from DataConfigImplementation?",
}

T = TypeVar("T")
_BUILTIN_TYPE_NAMES = ("String", "bool", "int", "float", "Vector2", "Vector3")


def main() -> None:
    generate_godot_code()
    generate_python_code()


def generate_godot_code() -> None:
    pass


def generate_python_code() -> None:
    spec_lines_by_name = load_files("specs")
    distribution_lines_by_name = load_files("utils/distributions")
    externally_defined_types = set(_BUILTIN_TYPE_NAMES + tuple(distribution_lines_by_name.keys()))
    parsed_classes = parse_classes(spec_lines_by_name, externally_defined_types)

    output = []
    output.extend(get_header_lines())

    output.extend(get_const_lines())

    for parsed_class in parsed_classes:
        output.extend(parsed_class.get_lines())

    output_file = Path(__file__).parent.joinpath("godot_generated_types.py")
    print(f"Writing generated types to {output_file}")  # script
    with open(output_file, "w") as out:
        out.write("\n".join(output).strip() + "\n")


def get_const_lines() -> List[str]:
    const_file = Path(__file__).parent.joinpath(f"godot/game/utils/caps/CONST.gd")
    const_lines = []
    with open(const_file, "r") as infile:
        for line in infile.readlines():
            if line.startswith("const "):
                if "[" not in line and "{" not in line:
                    declaration = line.replace(":=", "=").replace("const ", "").strip()
                    variable, value = re.split(r"\s*=\s+", declaration, 1)
                    const_lines.append(f"{variable}: Final = {value}")
    return const_lines + ["", ""]


class ParsedAttribute:
    def __init__(self, name: str, type_name: str, docstring: str, requirements: Set[str], is_optional: bool) -> None:
        self.name = name
        self.type_name = type_name
        self.docstring = docstring
        self.requirements = requirements
        self.is_optional = is_optional

    def get_line(self) -> str:
        if self.is_optional:
            type_name = f"Optional[{self.python_type_name}]"
        else:
            type_name = self.python_type_name
        line = f"    {self.name}: {type_name}"
        if self.docstring:
            line = f"    {self.docstring}\n{line}"
        return line

    @property
    def python_type_name(self) -> str:
        if self.type_name == "String":
            return "str"
        if self.type_name == "Vector2":
            return "AnyVec2"
        if self.type_name == "Vector3":
            return "AnyVec3"
        return self.type_name


class ParsedClass:
    def __init__(self, name: str, base: str, attributes: List[ParsedAttribute]) -> None:
        self.name = name
        self.base = base
        self.attributes = attributes

    def get_lines(self) -> List[str]:
        attribute_lines = [x.get_line() for x in self.attributes]
        if len(attribute_lines) == 0:
            attribute_lines.append("    pass")
        lines = [
            f"@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)",
            f"class {self.name}({self.python_base}):",
            *attribute_lines,
            "",
            "",
        ]
        comment = COMMENTS.get(self.name, None)
        if comment is not None:
            return [f"# {comment}", *lines]
        return lines

    @property
    def python_base(self) -> str:
        if self.name == "SimSpec":
            return "SpecBase, DataConfigImplementation"
        return self.base


def parse_classes(
    lines_by_name: Dict[str, List[str]],
    externally_defined_types: Set[str],
) -> List[ParsedClass]:
    lines_by_name.pop("SpecBase")
    generated_classes = [ParsedClass(name="SpecBase", base="Serializable", attributes=[])]
    required_names_by_class_name = {k: get_requirements(v, externally_defined_types) for k, v in lines_by_name.items()}
    while len(lines_by_name) > 0:
        available_class_names = {x.name for x in generated_classes}
        classes_to_generate = [
            x
            for x in lines_by_name.keys()
            if len(required_names_by_class_name[x].difference(available_class_names)) == 0
        ]
        if len(classes_to_generate) == 0:
            print("Dependencies:")  # script
            for class_name in lines_by_name:
                print(f"    {class_name}: {required_names_by_class_name[class_name]}")  # script
            raise Exception("Some classes do not inherit from SpecBase? " + str(list(lines_by_name.keys())))
        for class_name in sorted(classes_to_generate):
            generated_classes.append(parse_class(class_name, lines_by_name[class_name], externally_defined_types))
            lines_by_name.pop(class_name)
    return generated_classes


def get_requirements(lines: List[str], externally_defined_types: Set[str]) -> Set[str]:
    extends_line = only([x for x in lines if x.startswith("extends")])
    base = extends_line.split(" ")[-1]
    attributes = parse_attributes(lines, externally_defined_types)
    return {base}.union(flatten([x.requirements for x in attributes]))


def parse_attributes(lines: List[str], externally_defined_types: Set[str]) -> List[ParsedAttribute]:
    attributes = []
    next_docstring = []
    for line in lines:
        try:
            if line.startswith("#"):
                next_docstring.append(line)
            elif line.startswith("var "):
                name = line.split(":")[0].replace("var ", "")
                type_name = line.split(":")[1].strip()
                if "=" in type_name:
                    type_name = type_name.split("=")[0].strip()
                requirements = set()
                is_optional = "#optional" in type_name
                if is_optional:
                    type_name = type_name.split("#")[0].strip()
                elif "#" in type_name:
                    type_name = type_name.split("#")[0].strip() + type_name.split("#")[1]
                bare_type_name = type_name.split("[")[0]
                if not is_optional and bare_type_name not in externally_defined_types:
                    requirements.add(bare_type_name)
                docstring = "\n    ".join(next_docstring)
                attributes.append(
                    ParsedAttribute(
                        name=name,
                        type_name=type_name,
                        docstring=docstring,
                        requirements=requirements,
                        is_optional=is_optional,
                    )
                )
                next_docstring = []
        except Exception as e:
            raise Exception(f"Error parsing line: {line}") from e
    return attributes


def parse_class(name: str, lines: List[str], externally_defined_types: Set[str]) -> ParsedClass:
    extends_line = only([x for x in lines if x.startswith("extends")])
    base = extends_line.split(" ")[-1]
    return ParsedClass(name=name, base=base, attributes=parse_attributes(lines, externally_defined_types))


def get_header_lines() -> List[str]:
    return [
        "# GENERATED FILE",
        "# See generate_godot_code.py for details",
        "",
        "from typing import Final",
        "from typing import Optional",
        "",
        "import attr",
        "",
        "from avalon.datagen.godot_base_types import *",
        "",
        "",
    ]


def load_files(file_type_name: str) -> Dict[str, List[str]]:
    lines_by_name: Dict[str, List[str]] = {}
    folder = Path(__file__).parent.joinpath(f"godot/game")
    for spec_file in folder.glob(f"{file_type_name}/*.gd"):
        lines = spec_file.read_text().splitlines()
        lines = [x.rstrip() for x in lines if x.strip()]
        class_name_line = only([x for x in lines if x.startswith("class_name")])
        class_name_from_text = class_name_line.split(" ")[-1]
        lines_by_name[class_name_from_text] = lines
    if len(lines_by_name) == 0:
        raise Exception(f"Found no {file_type_name} files...")
    return lines_by_name


def flatten(iterable: Iterable[Iterable[T]]) -> List[T]:
    return list(itertools.chain(*iterable))


def only(x: Iterable[T]) -> T:
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


if __name__ == "__main__":
    main()
