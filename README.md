# Self-Building Program (Offline JVM Skeleton)

This repository is a Java 17, multi-module Gradle project that demonstrates an offline, self-evolving JVM application. A minimal trusted kernel ingests a `spec.json`, produces a deterministic plan and code bundle, verifies it, compiles it in-memory, loads it via a versioned classloader, runs tests and shadow execution, then atomically swaps the active implementation.

## Modules

- **api**: Stable interfaces and DTOs for classloader boundaries.
- **kernel**: Spec parsing, planning/codegen, verification, compilation, hot swapping, shadow execution, and registry persistence.
- **app**: Runnable entry point that drives the kernel loop.

## How to Run

```bash
./gradlew :app:run --args="spec.json ./config.json"
```

## How Iteration Works

The app runs up to three iterations:
1. Parse `spec.json` and compute a deterministic `specHash`.
2. Use a stub planner and code generator (seeded by `specHash`) to create sources in a new versioned package.
3. Verify source safety rules and compile in memory.
4. Verify bytecode constants for forbidden references.
5. Load the candidate into a new classloader.
6. Run in-process tests and invoke any `selfCheck()` hook.
7. Run a deterministic shadow corpus (examples + PRNG-synthesized inputs).
8. Switch to the candidate if all checks pass, otherwise record failure and iterate.

The first generated implementation intentionally violates one invariant for a deterministic "poison" input. The next iteration fixes it, demonstrating rollback and recovery behavior.

## Registry

The version registry is stored at:

```
./var/registry.json
```

Each entry records plan summaries, verification results, compilation diagnostics, test/shadow outcomes, and decisions.

## Deterministic Seed

`specHash` is the SHA-256 of the raw `spec.json` bytes. The hash seeds the planner, code generator, and the shadow corpus PRNG so that behavior is reproducible across runs.

## Adding Verifier Rules

Extend `Verifier` in the `kernel` module:
- Add new forbidden tokens to the hardcoded list.
- Enhance source scanning rules.
- Add new constant-pool checks.

## Replacing Stub Planner/CodeGen

The kernel now supports an offline LLM adapter backed by the local `inference4j` library. The stub planner/codegen remain the default fallback.

### Enable inference4j

1. Place a local GGUF model on disk (no downloads at runtime).
2. Update `config.json`:

```json
{
  "llm": {
    "mode": "inference4j",
    "inference4j": {
      "model_path": "/absolute/path/to/model.gguf",
      "context_length": 4096
    },
    "timeout_ms": 30000,
    "max_output_chars": 16000,
    "seed": 42,
    "temperature": 0.2,
    "max_tokens": 512,
    "top_p": 0.9
  }
}
```

3. Run:

```bash
./gradlew :app:run --args="spec.json ./config.json"
```

### Determinism and JSON-only output

- If `llm.seed` is omitted, the kernel derives a deterministic seed from `specHash`.
- The inference adapter enforces max output size, timeouts, and validates strict JSON-only responses for both planning and codegen.
- Any non-JSON output is rejected after extraction/validation.

### Troubleshooting

- **Timeouts**: increase `llm.timeout_ms`.
- **Invalid JSON**: lower temperature or force `temperature: 0.0`.
- **Verifier failures**: adjust constraints or ensure generated code only uses the stable API.
