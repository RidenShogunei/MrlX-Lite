"""Patch Plancraft's Windows tag-name parsing bug in the installed package.

Plancraft 0.4.9 parses tag filenames with ``tag_file.split("/")`` inside
``environment/recipes.py``. On Windows this keeps the full backslash path as the
tag name, so importing ``plancraft.environment`` fails with ``KeyError:
'acacia_logs'``. The patch is intentionally tiny and idempotent.
"""

from pathlib import Path

import plancraft


OLD = 'tag_name = tag_file.split("/")[-1].split(".")[0]'
NEW = "tag_name = os.path.splitext(os.path.basename(tag_file))[0]"


def main():
    package_dir = Path(plancraft.__file__).resolve().parent
    recipes_py = package_dir / "environment" / "recipes.py"
    text = recipes_py.read_text(encoding="utf-8")
    if NEW in text:
        print(f"[patch-plancraft-windows] already patched: {recipes_py}")
        return
    if OLD not in text:
        raise RuntimeError(f"Could not find expected Plancraft tag parsing line in {recipes_py}")
    recipes_py.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"[patch-plancraft-windows] patched: {recipes_py}")


if __name__ == "__main__":
    main()
