package com.llama4j;

import com.llama4j.tokenizer.Tokenizer;
import com.llama4j.tokenizer.Vocabulary;
import com.llama4j.util.Pair;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for the Tokenizer implementation.
 */
class TokenizerTest {
    /**
     * Verifies that encoding and decoding returns the original input.
     */
    @Test
    void encodeDecodeRoundTrip() {
        String[] tokens = new String[256];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = new String(Character.toChars(Tokenizer.BYTE_ENCODER.get(i)));
        }
        Vocabulary vocabulary = new Vocabulary(tokens, null);
        Tokenizer tokenizer = new Tokenizer(vocabulary, List.<Pair<Integer, Integer>>of(), ".", Map.of());

        String input = "Hello!";
        List<Integer> encoded = tokenizer.encodeAsList(input);
        String decoded = tokenizer.decode(encoded);

        assertEquals(input, decoded);
    }
}
