from __future__ import annotations

import os
import platform
import resource
import sys
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn


def apply_thread_limit(threads: int) -> None:
    threads = max(1, int(threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(threads)


def system_metadata(threads: int) -> dict[str, Any]:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_bytes = int(maxrss if sys.platform == "darwin" else maxrss * 1024)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_limit": int(threads),
        "max_rss_bytes": maxrss_bytes,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "representation_backend": "numpy-random-features",
    }
