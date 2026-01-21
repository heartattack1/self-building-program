package com.llama4j.config;

/**
 * Provides shared default values for the Llama runtime.
 */
public final class LlamaDefaults {
    /**
     * Default maximum number of tokens to generate when no override is provided.
     */
    public static final int DEFAULT_MAX_TOKENS = 512;

    private LlamaDefaults() {
    }
}
