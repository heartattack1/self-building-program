package com.llama4j.tokenizer;

import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Common tokenizer interface for model-specific tokenizers.
 */
public interface Tokenizer {
    /**
     * Returns the regex pattern used by the tokenizer.
     *
     * @return regex pattern string or null
     */
    String regexPattern();

    /**
     * Returns the map of special token strings to indices.
     *
     * @return special token map
     */
    Map<String, Integer> getSpecialTokens();

    /**
     * Returns true if the token index corresponds to a special token.
     *
     * @param tokenIndex token index
     * @return true if special token
     */
    boolean isSpecialToken(int tokenIndex);

    /**
     * Encodes a string into token ids using the tokenizer.
     *
     * @param text text to encode
     * @return token ids
     */
    int[] encode(String text);

    /**
     * Encodes a string into a list of token ids.
     *
     * @param text text to encode
     * @return list of token ids
     */
    List<Integer> encodeAsList(String text);

    /**
     * Encodes text while handling special tokens.
     *
     * @param text text to encode
     * @param allowedSpecial set of allowed special token strings
     * @return token ids
     */
    List<Integer> encode(String text, Set<String> allowedSpecial);

    /**
     * Encodes text while ignoring special tokens.
     *
     * @param text text to encode
     * @return token ids
     */
    List<Integer> encodeOrdinary(String text);

    /**
     * Decodes token ids back into a string.
     *
     * @param tokens token ids
     * @return decoded string
     */
    String decode(List<Integer> tokens);

    /**
     * Replaces control characters in code points with escaped sequences.
     *
     * @param codePoints code points to sanitize
     * @return sanitized string
     */
    static String replaceControlCharacters(int[] codePoints) {
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
            } else {
                chars.appendCodePoint(cp);
            }
        }
        return chars.toString();
    }

    /**
     * Replaces control characters in a string with escaped sequences.
     *
     * @param str string to sanitize
     * @return sanitized string
     */
    static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }
}
