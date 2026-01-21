package com.llama4j.tokenizer;

/**
 * Tokenizer provider for BERT models.
 */
public class BertTokenizerProvider implements TokenizerProvider {
    @Override
    public String modelName() {
        return "bert";
    }

    @Override
    public Tokenizer createTokenizer(TokenizerConfig config) {
        if (config.vocabulary() == null) {
            throw new IllegalArgumentException("BERT tokenizer requires a vocabulary");
        }
        String unknownToken = config.unknownToken() == null ? "[UNK]" : config.unknownToken();
        return new SimpleTokenizer(config.vocabulary(), config.specialTokens(), unknownToken);
    }
}
