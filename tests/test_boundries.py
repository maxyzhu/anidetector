"""The dependency arrows, as an executable rule.

    core  <-  inference  <-  image
      ^           ^
      +-----------+--------  video

Broken quietly, this is the single change that turns the two workflows into one
tangle, and nothing else would catch it until someone tried to lift a package
out or reuse it without Django.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# package -> top-level modules it must never import.
#
# inference forbids core as well as the rest: it ended up as the leaf of the
# graph, which is stronger than the arrow the refactor plan drew, and worth
# locking in while it is still true.
FORBIDDEN = {
    "core": ("image", "video", "inference"),
    "inference": ("core", "image", "video", "django"),
    "image": ("video",),
    "video": ("image",),
}

_IMPORT = re.compile(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", re.MULTILINE)


def _imports(path):
    for match in _IMPORT.finditer(path.read_text()):
        yield match.group(1) or match.group(2)


@pytest.mark.parametrize("package, forbidden", sorted(FORBIDDEN.items()))
def test_a_package_never_imports_across_the_arrows(package, forbidden):
    offenders = []
    for source in sorted((ROOT / package).rglob("*.py")):
        if "migrations" in source.parts:
            continue          # generated, and they reference apps by name not import
        for name in _imports(source):
            if name in forbidden:
                offenders.append(f"{source.relative_to(ROOT)} imports {name}")
    assert not offenders, "\n" + "\n".join(offenders)


def test_the_apps_with_models_are_installed():
    from django.conf import settings

    assert {"core", "image", "video"} <= set(settings.INSTALLED_APPS)


def test_inference_is_not_an_installed_app():
    from django.conf import settings

    # No models, no views, no migrations. Registering it would be a lie that
    # eventually justifies putting a model in it.
    assert "inference" not in settings.INSTALLED_APPS