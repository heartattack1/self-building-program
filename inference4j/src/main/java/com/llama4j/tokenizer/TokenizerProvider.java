package com.llama4j.tokenizer;

/**
 * Service provider for model-specific tokenizers.
 */
public interface TokenizerProvider {
    /**
     * Returns the model name handled by this provider.
     *
     * @return model name
     */
    String modelName();

    /**
     * Creates a tokenizer from the supplied configuration.
     *
     * @param config tokenizer configuration
     * @return tokenizer implementation
     */
    Tokenizer createTokenizer(TokenizerConfig config);
}
