"""pnanolocz_lib: AFM and HS-AFM data analysis tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pnanolocz_lib")
except PackageNotFoundError:
    __version__ = "0.0.0"
