"""Bootstrap access to the vendored official BFCL package.

The official repo at ``third_party/bfcl-official`` is a plain (uninstalled)
Python package, so we make ``bfcl_eval`` importable by putting the vendored
root on ``sys.path``. Nothing in the vendored tree is ever modified.

One import-time side effect of ``bfcl_eval`` needs redirecting: importing
``bfcl_eval.constants.eval_config`` creates ``result/``, ``score/`` and
``.file_locks/`` directories under ``BFCL_PROJECT_ROOT`` (which defaults to
the vendored repo itself). We point ``BFCL_PROJECT_ROOT`` at our results
directory *before* the first ``bfcl_eval`` import so the vendored directory
stays pristine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "bfcl-official"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "results" / "bfcl" / "official_artifacts"

_bootstrapped: Path | None = None


def bootstrap_official(official_root: str | os.PathLike | None = None) -> Path:
    """Make the vendored ``bfcl_eval`` package importable and return its root.

    Idempotent: subsequent calls return the root chosen by the first call.
    """
    global _bootstrapped
    if _bootstrapped is not None:
        return _bootstrapped

    root = Path(
        official_root
        or os.getenv("BFCL_OFFICIAL_ROOT", "").split("#", 1)[0].strip()
        or DEFAULT_OFFICIAL_ROOT
    )
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()

    if not (root / "bfcl_eval").is_dir():
        raise FileNotFoundError(
            f"Official BFCL repo not found at {root} (expected a 'bfcl_eval' package). "
            "Set BFCL_OFFICIAL_ROOT or vendor the repo at third_party/bfcl-official."
        )

    # Redirect bfcl_eval's generated artifacts away from the vendored tree.
    if "BFCL_PROJECT_ROOT" not in os.environ:
        DEFAULT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        os.environ["BFCL_PROJECT_ROOT"] = str(DEFAULT_ARTIFACT_ROOT)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    _bootstrapped = root
    return root
