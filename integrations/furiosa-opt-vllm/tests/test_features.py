"""The serving table is backed by feature-marked tests without importing device suites."""

import ast
from pathlib import Path


def test_covers_every_feature() -> None:
    """Every documented cell has a marked test, and every feature mark names a documented row."""
    directory = Path(__file__).resolve().parent
    section = (
        (directory.parents[2] / "docs/serving/index.md")
        .read_text()
        .split("## Feature support\n", 1)[1]
        .split("\n## ", 1)[0]
    )
    table = [line.split("|")[1:-1] for line in section.splitlines() if line.startswith("|")]
    column = [cell.strip() for cell in table[0]].index("vLLM")
    cells = {row[0].strip(): row[column].strip() for row in table[2:]}
    assert cells, "Feature support table has no cells"
    marks: set[str] = set()
    for path in sorted(directory.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        roots = [
            decorator
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name.startswith(("test_", "Test"))
            for decorator in node.decorator_list
        ] + [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
        ]
        for root in roots:
            for node in ast.walk(root):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "pytest.mark.feature":
                    assert len(node.args) == 1 and not node.keywords, (
                        f"Invalid feature mark: {path}"
                    )
                    argument = node.args[0]
                    assert isinstance(argument, ast.Constant) and isinstance(argument.value, str), (
                        path
                    )
                    marks.add(argument.value)
    missing = [f"vLLM: {name}: {cells[name]}" for name in sorted(cells.keys() - marks)]
    unknown = [f"Unknown feature mark: {name}" for name in sorted(marks - cells.keys())]
    assert not missing + unknown, "\n".join(missing + unknown)
