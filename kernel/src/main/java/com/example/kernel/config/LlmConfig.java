package com.example.kernel.config;

import java.nio.charset.StandardCharsets;

public record LlmConfig(
        LlmMode mode,
        Inference4jConfig inference4j,
        Integer timeoutMs,
        Integer maxOutputChars,
        Long seed,
        Double temperature,
        Integer maxTokens,
        Double topP
) {
    public static LlmConfig defaults() {
        return new LlmConfig(
                LlmMode.STUB,
                Inference4jConfig.defaults(),
                30_000,
                16_000,
                null,
                0.2,
                512,
                0.9
        );
    }

    public LlmConfig merge(LlmConfig override) {
        if (override == null) {
            return this;
        }
        return new LlmConfig(
                override.mode() != null ? override.mode() : mode(),
                inference4j().merge(override.inference4j()),
                override.timeoutMs() != null ? override.timeoutMs() : timeoutMs(),
                override.maxOutputChars() != null ? override.maxOutputChars() : maxOutputChars(),
                override.seed() != null ? override.seed() : seed(),
                override.temperature() != null ? override.temperature() : temperature(),
                override.maxTokens() != null ? override.maxTokens() : maxTokens(),
                override.topP() != null ? override.topP() : topP()
        );
    }

    public long seedFor(String specHash) {
        if (seed != null) {
            return seed;
        }
        byte[] bytes = specHash.getBytes(StandardCharsets.UTF_8);
        long derived = 0;
        for (byte value : bytes) {
            derived = derived * 31 + value;
        }
        return derived;
    }
}
