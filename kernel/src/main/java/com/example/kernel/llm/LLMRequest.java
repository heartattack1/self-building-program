package com.example.kernel.llm;

import java.util.Map;

public record LLMRequest(
        String task,
        String specHash,
        String prompt,
        int maxTokens,
        double temperature,
        long seed,
        Map<String, String> metadata
) {
}
