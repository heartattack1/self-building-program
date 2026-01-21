package com.llama4j.config;

/**
 * Base configuration for language models.
 */
public abstract class ModelConfiguration {
    /**
     * Returns the model identifier.
     *
     * @return model name
     */
    public abstract String modelName();

    /**
     * Returns the vocabulary size.
     *
     * @return vocabulary size
     */
    public abstract int vocabularySize();

    /**
     * Returns the context length.
     *
     * @return context length
     */
    public abstract int contextLength();
}
