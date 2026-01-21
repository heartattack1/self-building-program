package com.llama4j.tokenizer;

import java.util.Map;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Represents the tokenizer vocabulary and token lookup tables.
 *
 * @param tokens token strings
 * @param scores optional token scores
 * @param tokenToIndex mapping of token string to token index
 */
public record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    /**
     * Creates a vocabulary and builds the token index map.
     *
     * @param vocabulary token strings
     * @param scores optional scores
     */
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    /**
     * Returns the token string at the given index.
     *
     * @param tokenIndex token index
     * @return token string
     */
    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    /**
     * Looks up the index of a token string.
     *
     * @param token token string
     * @return optional token index
     */
    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    /**
     * Returns the number of tokens in the vocabulary.
     *
     * @return vocabulary size
     */
    public int size() {
        return tokens.length;
    }
}
