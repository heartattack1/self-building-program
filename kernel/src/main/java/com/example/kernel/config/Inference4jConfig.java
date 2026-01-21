package com.example.kernel.config;

public record Inference4jConfig(
        String modelPath,
        Integer contextLength
) {
    public static Inference4jConfig defaults() {
        return new Inference4jConfig(null, null);
    }

    public Inference4jConfig merge(Inference4jConfig override) {
        if (override == null) {
            return this;
        }
        return new Inference4jConfig(
                override.modelPath() != null ? override.modelPath() : modelPath(),
                override.contextLength() != null ? override.contextLength() : contextLength()
        );
    }
}
