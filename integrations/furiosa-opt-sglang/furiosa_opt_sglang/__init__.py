"""SGLang discovers the platform and worker through package entry points."""


def activate() -> str:
    """Return the platform class path for the `sglang.srt.platforms` entry point."""
    return "furiosa_opt_sglang.platform.Platform"


def register() -> None:
    """Install the extension table at SGLang startup."""
    from furiosa_opt_sglang.hooks import register

    register()
