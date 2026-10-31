"""Page variables read from the plugin manifests and the workspace, so versions have one owner."""

import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Any

# The SGLang flags a task needs, and the vLLM serve flags of an inventory row that SGLang spells
# differently. A task or a row flag missing here, or a length of -1, has no SGLang command.
SGLANG_TASK_ARGS = {
    "text": [],
    "embedding": ["--is-embedding", "--json-model-override-args", '{"is_matryoshka": true}'],
    "reranking": [],
}
SGLANG_FLAG = {"--max-model-len": "--context-length", "--kv-cache-dtype": "--kv-cache-dtype"}


def sglang_args(row: dict[str, Any]) -> list[str] | None:
    if row["task"] not in SGLANG_TASK_ARGS:
        return None
    args = list(SGLANG_TASK_ARGS[row["task"]])
    serve = row.get("serve", [])
    for flag, value in zip(serve[::2], serve[1::2], strict=False):
        if flag not in SGLANG_FLAG or value == "-1":
            return None
        args += [SGLANG_FLAG[flag], value]
    return None if len(serve) % 2 else args


def model_types(nav: list[Any]) -> dict[str, str]:
    """Each model page's model type: the nav group under its family that lists the page."""
    models = next(item["Models"] for item in nav if isinstance(item, dict) and "Models" in item)
    return {
        page: model_type
        for family in models
        if isinstance(family, dict)
        for groups in family.values()
        for group in groups
        if isinstance(group, dict)
        for model_type, pages in group.items()
        for page in pages
    }


def decoder_classes(package: Path) -> list[str]:
    """Each model class the engine's type stubs in `package` declare, as `module.Class`: the classes
    that derive from `Decoder`."""
    return [
        f"{stub.stem}.{node.name}"
        for stub in sorted(package.glob("*.pyi"))
        for node in ast.parse(stub.read_text()).body
        if isinstance(node, ast.ClassDef)
        and any(ast.unparse(base) == "engine.Decoder" for base in node.bases)
    ]


def define_env(env: Any) -> None:
    root = Path(env.project_dir)
    env.variables["model_types"] = model_types(env.conf["nav"])
    stubs = root / "crates/furiosa-opt-engine/python/furiosa_opt_engine"
    env.variables["decoder_classes"] = decoder_classes(stubs)
    for plugin in ("vllm", "sglang"):
        manifest = root / f"integrations/furiosa-opt-{plugin}/pyproject.toml"
        project = tomllib.loads(manifest.read_text())["project"]
        env.variables[f"{plugin}_project"] = project
        pin = next(d for d in project["dependencies"] if d.startswith(plugin))
        env.variables[f"{plugin}_version"] = re.search(r"(\d+\.\d+\.\d+)", pin).group(1)
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    env.variables["std_rev"] = workspace["workspace"]["dependencies"]["furiosa-opt-std"]["rev"]
    summary = root / "docs/theme/data/nightly.json"
    env.variables["nightly_run"] = json.loads(summary.read_text()) if summary.exists() else None
    env.macro(sglang_args)
