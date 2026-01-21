package com.llama4j.tokenizer;

import com.llama4j.util.Pair;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Configuration container for building tokenizers.
 */
public final class TokenizerConfig {
    private final Vocabulary vocabulary;
    private final List<Pair<Integer, Integer>> merges;
    private final String regexPattern;
    private final Map<String, Integer> specialTokens;
    private final String unknownToken;

    private TokenizerConfig(Builder builder) {
        this.vocabulary = builder.vocabulary;
        this.merges = builder.merges;
        this.regexPattern = builder.regexPattern;
        this.specialTokens = builder.specialTokens;
        this.unknownToken = builder.unknownToken;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Vocabulary vocabulary() {
        return vocabulary;
    }

    public List<Pair<Integer, Integer>> merges() {
        return merges == null ? List.of() : Collections.unmodifiableList(merges);
    }

    public String regexPattern() {
        return regexPattern;
    }

    public Map<String, Integer> specialTokens() {
        return specialTokens == null ? Map.of() : Collections.unmodifiableMap(specialTokens);
    }

    public String unknownToken() {
        return unknownToken;
    }

    /**
     * Builder for tokenizer configurations.
     */
    public static final class Builder {
        private Vocabulary vocabulary;
        private List<Pair<Integer, Integer>> merges;
        private String regexPattern;
        private Map<String, Integer> specialTokens;
        private String unknownToken;

        public Builder vocabulary(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
            return this;
        }

        public Builder merges(List<Pair<Integer, Integer>> merges) {
            this.merges = merges;
            return this;
        }

        public Builder regexPattern(String regexPattern) {
            this.regexPattern = regexPattern;
            return this;
        }

        public Builder specialTokens(Map<String, Integer> specialTokens) {
            this.specialTokens = specialTokens;
            return this;
        }

        public Builder unknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return this;
        }

        public TokenizerConfig build() {
            return new TokenizerConfig(this);
        }
    }
}
