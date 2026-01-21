package com.example.kernel.llm;

public record InferenceResult(
        String text,
        int generatedTokens
) {
}
