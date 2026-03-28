import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def _is_shadowed_workspace_diffusers(module) -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_diffusers_dir = repo_root / "diffusers"

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            return workspace_diffusers_dir in Path(module_file).resolve().parents
        except OSError:
            return False

    module_paths = getattr(module, "__path__", None) or []
    try:
        resolved_paths = [Path(path).resolve() for path in module_paths]
    except OSError:
        return False
    return workspace_diffusers_dir.resolve() in resolved_paths


def load_official_diffusers():
    existing_module = sys.modules.get("diffusers")
    if existing_module is not None and not _is_shadowed_workspace_diffusers(existing_module):
        return existing_module

    try:
        package_dist = distribution("diffusers")
    except PackageNotFoundError as error:
        raise ImportError(
            "The official 'diffusers' package is not installed. Install it with `pip install diffusers`."
        ) from error

    package_init = Path(package_dist.locate_file("diffusers/__init__.py")).resolve()
    if not package_init.exists():
        raise ImportError(
            f"Could not locate the installed diffusers package at {package_init}."
        )

    spec = importlib.util.spec_from_file_location(
        "diffusers",
        package_init,
        submodule_search_locations=[str(package_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for installed diffusers at {package_init}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["diffusers"] = module
    spec.loader.exec_module(module)
    return module