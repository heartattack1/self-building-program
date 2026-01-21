package com.llama4j.tokenizer;

/**
 * Tokenizer provider for GPT-2 models.
 */
public class Gpt2TokenizerProvider implements TokenizerProvider {
    @Override
    public String modelName() {
        return "gpt2";
    }

    @Override
    public Tokenizer createTokenizer(TokenizerConfig config) {
        if (config.vocabulary() == null) {
            throw new IllegalArgumentException("GPT-2 tokenizer requires a vocabulary");
        }
        String unknownToken = config.unknownToken() == null ? "<unk>" : config.unknownToken();
        return new SimpleTokenizer(config.vocabulary(), config.specialTokens(), unknownToken);
    }
}
