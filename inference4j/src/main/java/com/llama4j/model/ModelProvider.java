package com.llama4j.model;

/**
 * Service provider for model implementations.
 */
public interface ModelProvider {
    /**
     * Returns the model name handled by this provider.
     *
     * @return model name
     */
    String modelName();

    /**
     * Creates a new model instance.
     *
     * @return model instance
     */
    Model createModel();
}
