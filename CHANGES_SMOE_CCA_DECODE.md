# Summary of Changes for SMoE + CCA_Decode vLLM Support

This document summarizes all modifications made to support GRPO training with the **Zyphra-staging/smoe-midtrain_phase2_decay-30036** model and **CCA_Decode/ROCm vLLM** (version 0.1.x) in the verl codebase. Each section includes file path, line numbers, and code diffs where applicable.

---

## 1. New Run Script

**File:** `verl/run_smoe_midtrain_phase2_decay_30036_math.sh`

- **Created** (new file), based on `verl/examples/grpo_trainer/run_qwen2-7b_math.sh`.
- Model: `actor_rollout_ref.model.path=Zyphra-staging/smoe-midtrain_phase2_decay-30036`
- Experiment name: `smoe_midtrain_phase2_decay_30036`
- Data paths: `/home/kgoginen/tas/train/data/gsm8k` and `/home/kgoginen/tas/train/data/math` (train/test parquet).
- Duplicate `param_offload` / `optimizer_offload` removed.
- Unsupported `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True` removed (CCA_Decode vLLM does not support `--disable-mm-preprocessor-cache`).

*(Full script is the new file; no inline diff.)*

---

## 2. FSDP Wrap Policy for SMoE

**File:** `verl/verl/utils/fsdp_utils.py`

### 2a. `get_fsdp_wrap_policy` (lines 99–102)

**After** `fsdp_transformer_layer_cls_to_wrap = _get_attr(...)` and **before** `min_num_params`:

```python
    # SMoE uses SMoEDecoderATTLayer and SMoEDecoderMLPLayer; _no_split_modules may list "SMoEDecoderLayer" which does not exist
    model_config = getattr(module, "config", None)
    if model_config is not None and getattr(model_config, "model_type", None) == "smoe":
        fsdp_transformer_layer_cls_to_wrap = ["SMoEDecoderATTLayer", "SMoEDecoderMLPLayer"]
```

**Before (conceptually):** no `model_config` check; code used `default_transformer_cls_names_to_wrap` (which for SMoE is `["SMoEDecoderLayer"]`, a non-existent class).

---

### 2b. `apply_fsdp2` (lines 519–522)

**After** `fsdp_transformer_layer_cls_to_wrap = config.get(...)` and **before** `if isinstance(fsdp_transformer_layer_cls_to_wrap, str)`:

```python
    # SMoE uses SMoEDecoderATTLayer and SMoEDecoderMLPLayer; _no_split_modules may list "SMoEDecoderLayer" which does not exist
    model_config = getattr(model, "config", None)
    if model_config is not None and getattr(model_config, "model_type", None) == "smoe":
        fsdp_transformer_layer_cls_to_wrap = ["SMoEDecoderATTLayer", "SMoEDecoderMLPLayer"]
```

**Before (conceptually):** no SMoE override; same wrong `_no_split_modules` for SMoE.

---

## 3. vLLM Version Bypass (Unsupported Versions)

**File:** `verl/verl/third_party/vllm/__init__.py`

### 3a. Lines 56–67 (else branch when version not in 0.7.0+ and SGLang not available)

**Before:**

```python
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.7.0+"
        )
```

**After:**

```python
    if not is_sglang_available():
        # Bypass: allow unsupported vllm versions (e.g. ROCm/custom dev builds) if import succeeds
        try:
            from vllm import LLM
            from vllm.distributed import parallel_state
            vllm_version = package_version
            VLLM_SLEEP_LEVEL = 1
        except Exception:
            raise ValueError(
                f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
                f"vllm versions are 0.7.0+"
            )
```

---

## 4. FlexibleArgumentParser / get_tcp_uri Import Fallback

**File:** `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

### 4a. Lines 81–88 (else branch for vLLM ≤ 0.11.0)

**Before:**

```python
else:
    from vllm.utils import FlexibleArgumentParser, get_tcp_uri
```

**After:**

```python
else:
    try:
        from vllm.utils import FlexibleArgumentParser, get_tcp_uri
    except ImportError:
        # CCA_Decode / ROCm vllm and some forks do not re-export from vllm.utils
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.utils.network_utils import get_tcp_uri
```

---

## 5. ExternalZeroMQDistributedExecutor: execute_model and sample_tokens (All Versions)

**File:** `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

### 5a. Lines 127–146

**Before:** Methods `execute_model` and `sample_tokens` were defined only inside `if _VLLM_VERSION >= version.parse("0.12.0"):`, so for CCA_Decode 0.1.x they were missing and the engine core received `future = None`.

**After:** Always define on `ExternalZeroMQDistributedExecutor` (no version guard):

```python
    # Required for all vLLM versions (including CCA_Decode/0.1.x): base Executor
    # does not implement these; without them engine core gets future=None and
    # raises 'NoneType' has no attribute 'result'.
    def execute_model(self, scheduler_output: Any, non_block: bool = False) -> Any:
        output = self.collective_rpc("execute_model", args=(scheduler_output,))
        result = output[0]
        if non_block:
            f = Future()
            f.set_result(result)
            return f
        return result

    def sample_tokens(self, grammar_output: Any, non_block: bool = False) -> Any:
        output = self.collective_rpc("sample_tokens", args=(grammar_output,))
        result = output[0]
        if non_block:
            f = Future()
            f.set_result(result)
            return f
        return result
```

---

## 6. served_model_name Always Set (CLI args)

**File:** `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

### 6a. Lines 346–353 (replacing the previous prometheus-only block)

**Before:**

```python
        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name
```

**After:**

```python
        # Always set served_model_name: some vLLM forks (e.g. CCA_Decode) require it in config/metrics
        if self.config.prometheus.enable and self.config.prometheus.served_model_name:
            served_model_name = self.config.prometheus.served_model_name
        else:
            served_model_name = self.model_config.local_path or "model"
        if "/" in served_model_name:
            served_model_name = served_model_name.split("/")[-1]
        args["served_model_name"] = served_model_name
```

---

## 7. init_app_state Signature and args.served_model_name (run_server)

**File:** `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

### 7a. Lines 451–463 (inside `run_server`, before and including `init_app_state` call)

**Before:**

```python
        app = build_app(args)
        if _VLLM_VERSION > version.parse("0.11.0"):
            await init_app_state(engine_client, app.state, args)
        else:
            await init_app_state(engine_client, vllm_config, app.state, args)
```

**After:**

```python
        # Ensure args.served_model_name exists (some vLLM forks require it in init_app_state)
        if getattr(args, "served_model_name", None) is None:
            name = self.model_config.local_path or "model"
            setattr(args, "served_model_name", name.split("/")[-1] or "model")

        app = build_app(args)
        # init_app_state signature: newer vLLM/CCA_Decode use (engine_client, state, args);
        # older vLLM used (engine_client, vllm_config, state, args). Detect by second param name.
        init_sig = inspect.signature(init_app_state)
        params = list(init_sig.parameters.keys())
        if len(params) >= 2 and params[1] == "state":
            await init_app_state(engine_client, app.state, args)
        else:
            await init_app_state(engine_client, vllm_config, app.state, args)
```

---

## 8. SMoE FLOPs Counter

**File:** `verl/verl/utils/flops_counter.py`

### 8a. New function (lines 533–585): `_estimate_smoe_flops`

**Added** (after `_estimate_gpt_oss_flops`, before `_estimate_unknown_flops`):

```python
def _estimate_smoe_flops(config, tokens_sum, batch_seqlens, delta_time):
    """FLOPs for Zyphra-style SMoE: layer-wise MoE (only some layers have experts). Uses
    hidden_size, kv_channels (head_dim), ffn_hidden_size_list, smoe_layers, moe_router_topk.
    """
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    head_dim = getattr(config, "kv_channels", None) or getattr(
        config, "head_dim", hidden_size // num_attention_heads
    )
    moe_topk = getattr(config, "moe_router_topk", None) or getattr(config, "num_experts_per_tok", 1)

    # Layer-wise MoE: only layers with non-zero ffn_hidden_size have MoE MLP
    ffn_list = getattr(config, "ffn_hidden_size_list", None)
    if ffn_list is not None:
        num_moe_layers = sum(1 for x in ffn_list if x and x > 0)
        moe_sizes = [x for x in ffn_list if x and x > 0]
        moe_intermediate_size = int(moe_sizes[0]) if moe_sizes else (hidden_size * 4)
    else:
        num_moe_layers = num_hidden_layers
        moe_intermediate_size = getattr(config, "moe_intermediate_size", hidden_size * 4)

    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        smoe_layers = getattr(config, "smoe_layers", [])
        expert_counts = [x for x in smoe_layers if isinstance(x, (int, float)) and x > 0]
        num_experts = int(expert_counts[0]) if expert_counts else 8

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
    emd_and_lm_head_N = vocab_size * hidden_size * 2

    dense_N = attn_linear_N * num_hidden_layers + moe_mlp_N * num_moe_layers + emd_and_lm_head_N
    dense_N_flops = 6 * dense_N * tokens_sum

    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved
```

### 8b. ESTIMATE_FUNC (line 612)

**Added** one entry:

```python
    "smoe": _estimate_smoe_flops,
```

**Before:** no `"smoe"` key.  
**After:** `ESTIMATE_FUNC` ends with `"mimo": _estimate_qwen2_flops,` and `"smoe": _estimate_smoe_flops,`.

---

## File and line reference summary

| File | Lines / location | Change |
|------|------------------|--------|
| `verl/run_smoe_midtrain_phase2_decay_30036_math.sh` | (new file) | New GRPO script for SMoE; data paths; removed duplicate/unsupported flags. |
| `verl/verl/utils/fsdp_utils.py` | 99–102 | SMoE override in `get_fsdp_wrap_policy`. |
| `verl/verl/utils/fsdp_utils.py` | 519–522 | SMoE override in `apply_fsdp2`. |
| `verl/verl/third_party/vllm/__init__.py` | 56–67 | Try import for unsupported vLLM versions; set version and sleep level on success. |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 81–88 | Import fallback for FlexibleArgumentParser / get_tcp_uri. |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 127–146 | Always define `execute_model` and `sample_tokens` on ExternalZeroMQDistributedExecutor. |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 346–353 | Always set `args["served_model_name"]`. |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 451–463 | Set `args.served_model_name`; detect `init_app_state` signature and call with 3 or 4 args. |
| `verl/verl/utils/flops_counter.py` | 533–585 | New `_estimate_smoe_flops`. |
| `verl/verl/utils/flops_counter.py` | 612 | Add `"smoe": _estimate_smoe_flops` to `ESTIMATE_FUNC`. |

---

## Notes for deployment

- If you run from a different verl tree (e.g. `/home/kgoginen/verl/`), apply the same edits there or run from this workspace.
- CCA_Decode vLLM and SMoE (Zyphra) are the primary targets; compatibility with standard vLLM 0.7.0+ and existing model types is preserved where possible.
