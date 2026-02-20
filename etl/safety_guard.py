from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from etl.tinvest_client import enforce_sandbox_environment


FORBIDDEN_METHOD_CALLS = {
    "post_order",
    "cancel_order",
    "replace_order",
    "get_portfolio",
    "get_operations",
    "get_operations_by_cursor",
    "sandbox_pay_in",
    "post_stop_order",
    "cancel_stop_order",
    "get_orders",
}

FORBIDDEN_IMPORT_NAMES = {
    "OrdersService",
    "StopOrdersService",
    "OperationsService",
    "SandboxService",
}

EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "venv", "data", "dist", "build", ".idea"}


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.name.startswith("test_") or "tests" in path.parts:
            continue
        yield path


def _collect_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                called_name = node.func.id
            else:
                called_name = None

            if called_name in FORBIDDEN_METHOD_CALLS:
                violations.append(
                    f"{path}:{node.lineno} forbidden trading call '{called_name}()'"
                )

        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name.startswith("t_tech.invest"):
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORT_NAMES:
                        violations.append(
                            f"{path}:{node.lineno} forbidden import '{module_name}.{alias.name}'"
                        )

    return violations


def enforce_data_only_repository(root: Path | None = None) -> None:
    project_root = root or Path(__file__).resolve().parent.parent
    violations: list[str] = []
    for file_path in _iter_python_files(project_root):
        violations.extend(_collect_violations(file_path))

    if violations:
        violations_text = "\n".join(violations)
        raise RuntimeError(f"Data-only safety check failed:\n{violations_text}")


def run_startup_safety_checks(root: Path | None = None) -> None:
    enforce_sandbox_environment()
    enforce_data_only_repository(root=root)
