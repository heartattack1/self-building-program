package com.example.kernel.config;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import java.util.Locale;

public enum LlmMode {
    STUB,
    INFERENCE4J;

    @JsonCreator
    public static LlmMode fromString(String value) {
        if (value == null) {
            return STUB;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "stub" -> STUB;
            case "inference4j" -> INFERENCE4J;
            default -> throw new IllegalArgumentException("Unknown llm.mode: " + value);
        };
    }

    @JsonValue
    public String toJson() {
        return switch (this) {
            case STUB -> "stub";
            case INFERENCE4J -> "inference4j";
        };
    }
}
