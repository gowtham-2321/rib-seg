"""Patch the pip-installed `straug` so it runs on NumPy 2 and recent scikit-image.

  * np.float_ / np.float / np.int were removed in NumPy 2.0
  * skimage.util.random_noise no longer accepts `seed=`

Safe to run repeatedly (idempotent).
"""
import importlib.util
import os
import re

spec = importlib.util.find_spec("straug")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("straug is not installed - run `pip install --no-deps straug` first")
root = list(spec.submodule_search_locations)[0]

alias_re = re.compile(r"\bnp\.(float_|float|int)\b")
seed_re = re.compile(r"(random_noise\([^\n]*?),\s*seed=\w+")


def alias_sub(m):
    return "int" if m.group(1) == "int" else "np.float64"


changed = 0
for name in sorted(os.listdir(root)):
    if not name.endswith(".py"):
        continue
    path = os.path.join(root, name)
    text = open(path, newline="").read()
    new = alias_re.sub(alias_sub, text)
    if name == "noise.py":
        new = seed_re.sub(r"\1", new)
    if new != text:
        open(path, "w", newline="").write(new)
        print("patched", name)
        changed += 1

print(f"straug patch: {changed} file(s) changed" if changed else "straug patch: already up to date")