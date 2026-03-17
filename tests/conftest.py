import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_rich_stubs():
    rich_module = types.ModuleType("rich")
    console_module = types.ModuleType("rich.console")
    panel_module = types.ModuleType("rich.panel")
    rule_module = types.ModuleType("rich.rule")
    syntax_module = types.ModuleType("rich.syntax")
    text_module = types.ModuleType("rich.text")

    class DummyConsole:
        def print(self, *args, **kwargs):
            pass

    class DummyPanel:
        def __init__(self, *args, **kwargs):
            pass

    class DummySyntax:
        def __init__(self, *args, **kwargs):
            pass

    class DummyText:
        def __init__(self, *args, **kwargs):
            pass

    class DummyRule:
        def __init__(self, *args, **kwargs):
            pass

    console_module.Console = DummyConsole
    panel_module.Panel = DummyPanel
    rule_module.Rule = DummyRule
    syntax_module.Syntax = DummySyntax
    text_module.Text = DummyText
    rich_module.box = types.SimpleNamespace(ROUNDED="ROUNDED")

    sys.modules.setdefault("rich", rich_module)
    sys.modules["rich.console"] = console_module
    sys.modules["rich.panel"] = panel_module
    sys.modules["rich.rule"] = rule_module
    sys.modules["rich.syntax"] = syntax_module
    sys.modules["rich.text"] = text_module


_install_rich_stubs()
