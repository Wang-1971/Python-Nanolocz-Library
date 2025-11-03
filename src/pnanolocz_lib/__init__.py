"""pnanolocz_lib: AFM and HS-AFM data analysis tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except Exception:
    try:
        __version__ = version("pnanolocz_lib")
    except PackageNotFoundError:
        __version__ = "0.0.0"
