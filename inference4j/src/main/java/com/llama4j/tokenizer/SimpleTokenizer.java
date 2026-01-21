package com.llama4j.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Simple whitespace tokenizer for demo and test models.
 */
public class SimpleTokenizer implements Tokenizer {
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final String unknownToken;

    /**
     * Creates a simple tokenizer with the given vocabulary.
     *
     * @param vocabulary vocabulary to use
     * @param specialTokens special token map
     * @param unknownToken token to use for unknown inputs
     */
    public SimpleTokenizer(Vocabulary vocabulary, Map<String, Integer> specialTokens, String unknownToken) {
        this.vocabulary = vocabulary;
        this.specialTokens = Map.copyOf(specialTokens);
        this.unknownToken = unknownToken;
    }

    @Override
    public String regexPattern() {
        return null;
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    @Override
    public int[] encode(String text) {
        return encodeAsList(text).stream().mapToInt(i -> i).toArray();
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text, Set.of());
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        if (allowedSpecial != null && !allowedSpecial.isEmpty() && specialTokens.keySet().contains(text)) {
            return List.of(specialTokens.get(text));
        }
        return encodeOrdinary(text);
    }

    @Override
    public List<Integer> encodeOrdinary(String text) {
        List<Integer> ids = new ArrayList<>();
        String trimmed = text.trim();
        if (trimmed.isEmpty()) {
            return ids;
        }
        String[] parts = trimmed.split("\\s+");
        for (String part : parts) {
            int token = vocabulary.getIndex(part).orElseGet(() -> vocabulary.getIndex(unknownToken).orElseThrow());
            ids.add(token);
        }
        return ids;
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(vocabulary.get(tokens.get(i)));
        }
        return sb.toString();
    }
}
