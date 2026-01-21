package com.llama4j.tokenizer;

/**
 * Tokenizer provider for Llama models.
 */
public class LlamaTokenizerProvider implements TokenizerProvider {
    @Override
    public String modelName() {
        return "llama";
    }

    @Override
    public Tokenizer createTokenizer(TokenizerConfig config) {
        if (config.vocabulary() == null) {
            throw new IllegalArgumentException("Llama tokenizer requires a vocabulary");
        }
        return new BpeTokenizer(config.vocabulary(), config.merges(), config.regexPattern(), config.specialTokens());
    }
}
