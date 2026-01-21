# Plan: Add Llama 3.2 1B Instruct GGUF Q4_K_M support (inference4j)

## Current state (analysis)

- The inference4j README states GGUF support is limited to `Q4_0`, `Q8_0`, `F16`, and `BF16`; other formats (including `Q5_K`) are explicitly not supported. The examples also reference `Q4_0` and `Q8_0` downloads only.【F:inference4j/README.md†L37-L83】
- The loader only instantiates `Q4_0FloatTensor`, `Q8_0FloatTensor`, `F16FloatTensor`, or `BF16FloatTensor` and throws for other GGML types; there is no path for `Q4_K` tensors today.【F:inference4j/src/main/java/com/llama4j/model/ModelLoader.java†L213-L227】
- The GGML type enum already defines `Q4_K` sizing constants, but there is no corresponding tensor implementation under `com.llama4j.tensor` (only `Q4_0` and `Q8_0`).【F:inference4j/src/main/java/com/llama4j/gguf/GGMLType.java†L6-L41】【F:inference4j/src/main/java/com/llama4j/tensor/Q4_0FloatTensor.java†L1-L139】

## Goal

Enable inference4j to load and run Llama 3.2 1B Instruct GGUF models quantized as `Q4_K_M` (GGML type `Q4_K`) while preserving existing `Q4_0/Q8_0/F16/BF16` functionality.

## Implementation plan

1. **Confirm GGUF tensor types in a Q4_K_M model**
   - Inspect a sample `Llama-3.2-1B-Instruct-Q4_K_M.gguf` file metadata using existing GGUF parsing utilities to confirm the tensor `GGMLType` values are reported as `Q4_K` and to check any metadata flags that distinguish `Q4_K_M` vs `Q4_K_S` (if relevant to scaling). This validates whether only tensor decoding changes are required.

2. **Implement `Q4_K` tensor decoding**
   - Add a new `Q4_KFloatTensor` in `com.llama4j.tensor` mirroring the `FloatTensor` API. It should:
     - Implement `getFloat` and `dot` for `Q4_K` blocks based on GGML’s Q4_K layout (block size 256, per-block scales, and per-16 sub-block mins; packed 4-bit values).
     - Prefer a vectorized path similar to `Q4_0FloatTensor.vectorDot` for performance, with a correct scalar fallback for correctness.
   - Source layout reference: `Q4_0FloatTensor` provides a working pattern for reading quantized blocks and implementing dot products in this codebase.【F:inference4j/src/main/java/com/llama4j/tensor/Q4_0FloatTensor.java†L1-L139】

3. **Wire `Q4_K` into the loader**
   - Update `ModelLoader.loadQuantized` to recognize `GGMLType.Q4_K` and return the new `Q4_KFloatTensor`.
   - Update error messaging to include `Q4_K` as supported.

4. **Update README and examples**
   - Expand the supported quantization section to include `Q4_K` (and explicitly note that `Q4_K_M` uses `Q4_K` tensors) with a recommended download link for Llama 3.2 1B Instruct Q4_K_M.

5. **Add tests / validation**
   - Add a lightweight unit test (or diagnostic) that loads a small GGUF tensor entry encoded as Q4_K and verifies `getFloat`/`dot` outputs against a known reference (can be synthesized or derived from a known-good implementation).
   - If a full model isn’t available in CI, add a local-only validation script or test profile that is skipped by default but documents the steps for manual verification.

6. **Integration check**
   - Run the inference4j CLI with a Q4_K_M model (Llama 3.2 1B Instruct) to confirm tokenizer and model execution succeed without `UnsupportedOperationException`.

## Risks / open questions

- `Q4_K_M` vs `Q4_K_S` are both encoded as `Q4_K` tensor types; if the GGUF metadata influences scaling behavior, ensure the tensor decoding aligns with ggml’s `q4_k` reference implementation.
- Performance tuning may be needed once correctness is confirmed; start with a correct scalar implementation if vectorization is complex.
