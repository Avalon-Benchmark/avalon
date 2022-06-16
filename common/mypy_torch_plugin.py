import subprocess
from typing import List
from typing import Set

from mypy.plugin import MethodSigContext
from mypy.plugin import Plugin
from mypy.types import CallableType
from mypy.types import Instance

LOCAL_CLASS_NAMES: Set[str] = set()


class TorchForwardInferencePlugin(Plugin):
    def get_method_signature_hook(self, fullname: str):
        if fullname.startswith("torch.nn.modules.") and fullname.endswith(".__call__"):
            return fix_call_method
        if fullname in LOCAL_CLASS_NAMES:
            return fix_call_method
        return None


def fix_call_method(ctx: MethodSigContext) -> CallableType:
    caller = ctx.type
    if isinstance(caller, Instance):
        if "forward" in caller.type.names:
            # go get the forward method instead
            forward_method = caller.type.names["forward"].type
            if forward_method and isinstance(forward_method, CallableType):
                if caller.type.has_base("pytorch_lightning.core.lightning.LightningModule"):
                    return forward_method
                else:
                    # and trim the "self" argument off of it
                    return forward_method.copy_modified(
                        arg_types=forward_method.arg_types[1:],
                        arg_kinds=forward_method.arg_kinds[1:],
                        arg_names=forward_method.arg_names[1:],
                    )
    return ctx.default_signature


# noinspection PyUnusedLocal
def plugin(version: str):
    global LOCAL_CLASS_NAMES

    # I'm a bad person. This is a terrible way of getting all of the classes that define a forward method...
    class_lines = subprocess.getoutput(
        'grep -E -r --include="*.py" "^(    def forward|class )" . | grep "    def forward" -B 1 | grep "class " | grep -v "mypy_torch_plugin.py"'
    ).split("\n")
    classes: List[str] = []
    for class_line in class_lines:
        try:
            file_path, class_def = class_line.split(".py:class ")
        except ValueError:
            # Sometimes that grep command gets stuff that doesn't match
            continue
        module_path = file_path[2:].replace("/", ".")
        class_name = class_def.split("(")[0]
        classes.append(f"{module_path}.{class_name}")

    LOCAL_CLASS_NAMES = {f"{x}.__call__" for x in classes}

    return TorchForwardInferencePlugin
