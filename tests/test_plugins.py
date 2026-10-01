"""Tests for third-party layer and grid discovery through entry points.

Plugins load when nzcvm is first imported, so each test runs in a fresh
interpreter with a fake plugin distribution on ``PYTHONPATH``.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PLUGIN_FILES = {
    "demo_plugin/__init__.py": "",
    "demo_plugin/config.py": """
        from dataclasses import dataclass

        from nzcvm.config.grids import GridConfig
        from nzcvm.config.layers import LayerConfig


        @dataclass
        class DemoLayerConfig(LayerConfig):
            type: str = "demo"


        @dataclass
        class DemoGridConfig(GridConfig):
            type: str = "demo_grid"
        """,
    "demo_plugin/layer.py": """
        from nzcvm.layers.core import Layer

        from demo_plugin.config import DemoLayerConfig


        class DemoLayer(Layer[DemoLayerConfig], config_cls=DemoLayerConfig):
            def __call__(self, grid, model_range=None):
                return self.next_layer(grid)
        """,
    "demo_plugin/grid.py": """
        from nzcvm.grids.builder import build_grids_from_config

        from demo_plugin.config import DemoGridConfig


        @build_grids_from_config.register
        def build_demo(config: DemoGridConfig):
            return {}
        """,
    "demo_plugin-0.1.dist-info/METADATA": """
        Metadata-Version: 2.1
        Name: demo-plugin
        Version: 0.1
        """,
    "demo_plugin-0.1.dist-info/top_level.txt": "demo_plugin\n",
    "demo_plugin-0.1.dist-info/entry_points.txt": """
        [nzcvm.layer_configs]
        demo = demo_plugin.config

        [nzcvm.layer_impls]
        demo = demo_plugin.layer

        [nzcvm.grid_configs]
        demo = demo_plugin.config

        [nzcvm.grid_impls]
        demo = demo_plugin.grid
        broken = demo_plugin.missing
        """,
}


@pytest.fixture(scope="module")
def plugin_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("plugins")
    for name, contents in PLUGIN_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(contents).lstrip())
    return root


def run_with_plugin(plugin_path: Path, code: str) -> subprocess.CompletedProcess:
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            [str(plugin_path), os.environ.get("PYTHONPATH", "")]
        ),
        "COLUMNS": "200",
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_plugin_layer_resolves_from_config(plugin_path: Path) -> None:
    result = run_with_plugin(
        plugin_path,
        """
        from nzcvm.config.layers import LayerConfig
        from nzcvm.layers.core import layer_from_config

        config = LayerConfig.from_dict({"type": "demo"})
        print(layer_from_config(config).__qualname__)
        """,
    )
    assert result.stdout.strip() == "DemoLayer"


def test_plugin_grid_resolves_from_config(plugin_path: Path) -> None:
    result = run_with_plugin(
        plugin_path,
        """
        from nzcvm.config.grids import GridConfig
        from nzcvm.grids.builder import build_grids_from_config

        config = GridConfig.from_dict({"type": "demo_grid"})
        print(build_grids_from_config.dispatch(type(config)).__name__)
        """,
    )
    assert result.stdout.strip() == "build_demo"


def test_broken_plugin_is_recorded_not_raised(plugin_path: Path) -> None:
    result = run_with_plugin(
        plugin_path,
        """
        import nzcvm.grids
        from nzcvm import plugins

        for status in plugins.failed_plugins():
            print(status.entry_point.name, type(status.error).__name__)
        """,
    )
    assert result.stdout.strip() == "broken ModuleNotFoundError"
    assert "Failed to load nzcvm.grid_impls plugin 'broken'" in result.stderr


def test_cli_lists_builtin_and_plugin_types(plugin_path: Path) -> None:
    result = run_with_plugin(
        plugin_path,
        """
        from nzcvm.scripts.nzcvm_cli import app

        app(["plugins"])
        """,
    )
    rows = {
        tuple(cell.strip() for cell in line.split("│")[1:-1])
        for line in result.stdout.splitlines()
        if line.startswith("│")
    }
    assert ("layer", "clamp", "nzcvm.layers.clamp.ClampLayer", "builtin") in rows
    assert ("grid", "sw4", "nzcvm.grids.sw4.build_sw4", "builtin") in rows
    assert (
        "layer",
        "demo",
        "demo_plugin.layer.DemoLayer",
        "demo-plugin 0.1",
    ) in rows
    assert ("grid", "demo_grid", "demo_plugin.grid.build_demo", "demo-plugin 0.1") in (
        rows
    )
    assert "broken" in result.stdout
