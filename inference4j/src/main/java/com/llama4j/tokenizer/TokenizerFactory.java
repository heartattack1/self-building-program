package com.llama4j.tokenizer;

import java.util.Locale;
import java.util.ServiceLoader;

/**
 * Factory for creating model-specific tokenizers.
 */
public final class TokenizerFactory {
    private TokenizerFactory() {
    }

    /**
     * Creates a tokenizer for the given model name.
     *
     * @param modelName model identifier
     * @param config tokenizer configuration
     * @return tokenizer instance
     */
    public static Tokenizer createTokenizer(String modelName, TokenizerConfig config) {
        TokenizerProvider provider = loadProvider(modelName);
        return provider.createTokenizer(config);
    }

    /**
     * Creates a tokenizer with an empty configuration.
     *
     * @param modelName model identifier
     * @return tokenizer instance
     */
    public static Tokenizer createTokenizer(String modelName) {
        return createTokenizer(modelName, TokenizerConfig.builder().build());
    }

    private static TokenizerProvider loadProvider(String modelName) {
        ServiceLoader<TokenizerProvider> loader = ServiceLoader.load(TokenizerProvider.class);
        String normalized = modelName.toLowerCase(Locale.ROOT);
        for (TokenizerProvider provider : loader) {
            if (provider.modelName().equalsIgnoreCase(normalized)) {
                return provider;
            }
        }
        throw new IllegalArgumentException("Unknown tokenizer provider: " + modelName);
    }
}
