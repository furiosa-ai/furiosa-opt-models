"""vLLM and vLLM-Omni platform plugin entry points."""


def register() -> str:
    # vLLM imports this entry point during initialization, so importing vLLM here can cycle.
    return "furiosa_opt_vllm.platform.FuriosaPlatform"


def register_omni() -> str:
    return "furiosa_opt_vllm.omni.platform.Platform"
