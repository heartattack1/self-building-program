package com.example.kernel.llm;

public interface InferenceBackend {
    InferenceResult generate(GenerationRequest request);
}
