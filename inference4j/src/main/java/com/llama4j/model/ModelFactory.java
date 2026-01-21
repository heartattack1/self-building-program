package com.llama4j.model;

import java.util.Locale;
import java.util.ServiceLoader;

/**
 * Factory for creating model instances by name.
 */
public final class ModelFactory {
    private ModelFactory() {
    }

    /**
     * Creates a model instance for the given name.
     *
     * @param modelName model identifier
     * @return model instance
     */
    public static Model createModel(String modelName) {
        ModelProvider provider = loadProvider(modelName);
        return provider.createModel();
    }

    private static ModelProvider loadProvider(String modelName) {
        ServiceLoader<ModelProvider> loader = ServiceLoader.load(ModelProvider.class);
        String normalized = modelName.toLowerCase(Locale.ROOT);
        for (ModelProvider provider : loader) {
            if (provider.modelName().equalsIgnoreCase(normalized)) {
                return provider;
            }
        }
        throw new IllegalArgumentException("Unknown model provider: " + modelName);
    }
}
