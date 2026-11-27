# Disaggregated KV transfer

NIXL and Mooncake TCP move KV pages from a prefill server to a decode server; the framework references own accepted options and startup refusals, while this page gives paired-role recipes.

[Integrating frameworks](../engine/frameworks.md#disaggregated-kv-transfer) explains the shared bounded staging path.

## Compatibility

Both adapters retain their framework's native transfer protocol and use matching optional dependency bounds. The plugin package metadata in `integrations/furiosa-opt-vllm/pyproject.toml` and `integrations/furiosa-opt-sglang/pyproject.toml` is the source of truth for exact version specifiers. NIXL and Mooncake follow vLLM 0.31.0's [connector requirements](https://github.com/vllm-project/vllm/blob/v0.31.0/requirements/kv_connectors.txt): NIXL uses its pinned version, while Mooncake uses vLLM's minimum and excludes the next minor API line. SGLang 0.5.22's [project metadata](https://github.com/sgl-project/sglang/blob/v0.5.22/python/pyproject.toml) adds no NIXL or Mooncake package bound.

Mooncake is supported over TCP; RDMA remains deferred because its registration path rejects the engine-pinned staging allocation. NIXL transport selection remains native, and this integration does not certify every transport. Neither native control plane bridges vLLM to SGLang. Updating either framework pin requires reviewing its registration, control and completion lifecycle, then running that framework's adapter suite in [Testing](../develop/testing.md#run-the-framework-tests). Updating NIXL or Mooncake requires checking both adapters and running both suites; these suites do not certify transfer performance.

## Mooncake over TCP

### vLLM

For vLLM, install `furiosa-opt-vllm[mooncake]` for the bounded non-CUDA transfer engine and select the plugin connector on both servers. Give each server a distinct `engine_id`; requests must include vLLM's Mooncake `kv_transfer_params` (`remote_engine_id`, `remote_bootstrap_addr`, and `transfer_id`) to pair a producer and consumer:

```bash
FURIOSA_VISIBLE_CHIPS=0 vllm serve Qwen/Qwen3-0.6B --port 30000 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"FuriosaMooncakeConnector","kv_connector_module_path":"furiosa_opt_vllm.mooncake","kv_role":"kv_producer","engine_id":"prefill","kv_connector_extra_config":{"mooncake_protocol":"tcp"}}'
FURIOSA_VISIBLE_CHIPS=1 vllm serve Qwen/Qwen3-0.6B --port 30001 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"FuriosaMooncakeConnector","kv_connector_module_path":"furiosa_opt_vllm.mooncake","kv_role":"kv_consumer","engine_id":"decode","kv_connector_extra_config":{"mooncake_protocol":"tcp"}}'
```

The distinct `engine_id`s pair the servers through vLLM's `kv_transfer_params`. The [vLLM reference](vllm.md#disaggregated-serving) documents staging capacity, pool pressure and supported cache constraints.

### SGLang

Run the roles on separate chips:

```bash
SGLANG_DISAGG_STAGING_BUFFER=1 FURIOSA_VISIBLE_CHIPS=0 python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 8998
SGLANG_DISAGG_STAGING_BUFFER=1 FURIOSA_VISIBLE_CHIPS=1 python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30001 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp
```

Install `furiosa-opt-sglang[mooncake]`. Send each `/generate` request to both servers with the same `bootstrap_host` and `bootstrap_port` (`8998` above), and a unique `bootstrap_room`; the decode server returns generated tokens. [SGLang's disaggregation reference](sglang.md#disaggregated-serving) documents its staging lifecycle and runtime constraints.

## NIXL

### vLLM

Install `furiosa-opt-vllm[nixl]`; its dependency bound is declared in the package metadata described under [Compatibility](#compatibility). With vLLM, a prefill server computes a prompt's cache and a decode server continues the request from it, through vLLM's NIXL connectors (`NixlConnector`, which names `NixlPullConnector`, or `NixlPushConnector`) in their host-buffer mode:

```bash
vllm serve Qwen/Qwen3-0.6B --port 8100 \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cpu"}'
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve Qwen/Qwen3-0.6B --port 8200 \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cpu"}'
```

Route a request to prefill with `"kv_transfer_params": {"do_remote_decode": true}` and `max_tokens` 1, then send the returned `kv_transfer_params` to decode under the same request ID (`X-Request-Id` or `request_id`). `NixlPushConnector` pairs the decode with its prefill by that ID. The [vLLM reference](vllm.md#disaggregated-serving) documents staging, connector composition and runtime constraints.

### SGLang

Install `furiosa-opt-sglang[nixl]` and set `SGLANG_DISAGG_STAGING_BUFFER=1` on both roles:

```bash
SGLANG_DISAGG_STAGING_BUFFER=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl

SGLANG_DISAGG_STAGING_BUFFER=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl
```

This uses SGLang's native NIXL network and bootstrap settings. The [SGLang reference](sglang.md#disaggregated-serving) documents its bounded staging behavior and framework-specific limits.
