package com.llama4j.tokenizer;

import com.llama4j.util.Pair;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class TokenizerFactoryTest {
    @Test
    void createsLlamaTokenizerFromConfig() {
        Vocabulary vocabulary = new Vocabulary(new String[]{"a", "b", "ab"}, null);
        TokenizerConfig config = TokenizerConfig.builder()
                .vocabulary(vocabulary)
                .merges(List.of(new Pair<>(0, 1)))
                .regexPattern(".+")
                .specialTokens(Map.of())
                .build();

        Tokenizer tokenizer = TokenizerFactory.createTokenizer("llama", config);

        assertInstanceOf(BpeTokenizer.class, tokenizer);
        assertEquals(List.of(2), tokenizer.encodeAsList("ab"));
        assertEquals("ab", tokenizer.decode(List.of(2)));
    }

    @Test
    void createsSimpleTokenizerForGpt2() {
        Vocabulary vocabulary = new Vocabulary(new String[]{"<unk>", "hello"}, null);
        TokenizerConfig config = TokenizerConfig.builder()
                .vocabulary(vocabulary)
                .unknownToken("<unk>")
                .build();

        Tokenizer tokenizer = TokenizerFactory.createTokenizer("gpt2", config);

        assertEquals(List.of(1), tokenizer.encodeAsList("hello"));
    }

    @Test
    void createsSimpleTokenizerForBert() {
        Vocabulary vocabulary = new Vocabulary(new String[]{"[UNK]", "hi"}, null);
        TokenizerConfig config = TokenizerConfig.builder()
                .vocabulary(vocabulary)
                .unknownToken("[UNK]")
                .build();

        Tokenizer tokenizer = TokenizerFactory.createTokenizer("bert", config);

        assertEquals(List.of(1), tokenizer.encodeAsList("hi"));
    }
}
