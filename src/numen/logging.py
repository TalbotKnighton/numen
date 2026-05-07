from __future__ import annotations

import logging

_FMT = "%(asctime)s [%(name)-28s] %(levelname)-8s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int = logging.DEBUG) -> None:
    """Enable numen framework logging to stderr.

    Surfaces backend diagnostics, solve timings, and Julia output in real time.
    Call once at the top of your script or notebook.

    Example::

        import logging
        from numen.logging import configure_logging
        configure_logging(level=logging.DEBUG)   # everything
        configure_logging(level=logging.INFO)    # solve start/finish only
    """
    root = logging.getLogger("numen")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
