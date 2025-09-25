"""Bull Machine version information - Single Source of Truth."""

__version__ = "1.5.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Version metadata
VERSION_MAJOR = __version_info__[0]
VERSION_MINOR = __version_info__[1]
VERSION_PATCH = __version_info__[2]

def get_version() -> str:
    """Get the current version string."""
    return __version__

def get_version_banner() -> str:
    """Get formatted version banner for CLI/logs."""
    return f"Bull Machine v{__version__}"

def get_version_info() -> dict:
    """Get detailed version information."""
    return {
        "version": __version__,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "version_tuple": __version_info__
    }