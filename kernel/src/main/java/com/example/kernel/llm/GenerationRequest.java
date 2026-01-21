package com.example.kernel.llm;

public record GenerationRequest(
        String prompt,
        int maxTokens,
        double temperature,
        double topP,
        long seed
) {
}
