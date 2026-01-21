package com.example.kernel.config;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

public record KernelConfig(LlmConfig llm) {
    private static final Logger LOGGER = Logger.getLogger(KernelConfig.class.getName());

    public static KernelConfig defaults() {
        return new KernelConfig(LlmConfig.defaults());
    }

    public static KernelConfig load(Path path) {
        KernelConfig defaults = defaults();
        if (path == null || Files.notExists(path)) {
            if (path != null) {
                LOGGER.info(() -> "Config file not found, using defaults: " + path);
            }
            return defaults;
        }
        ObjectMapper mapper = new ObjectMapper();
        mapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        try {
            KernelConfig loaded = mapper.readValue(Files.readAllBytes(path), KernelConfig.class);
            return new KernelConfig(defaults.llm().merge(loaded.llm()));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load config: " + path, e);
        }
    }
}
