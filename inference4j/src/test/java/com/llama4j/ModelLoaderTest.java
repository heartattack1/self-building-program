package com.llama4j;

import com.llama4j.model.Llama;
import com.llama4j.model.ModelLoader;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Tests for the ModelLoader factory.
 */
class ModelLoaderTest {
    /**
     * Ensures model construction from metadata is successful.
     */
    @Test
    void buildsModelFromMetadata() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("tokenizer.ggml.model", "gpt2");

        int baseTokens = 128000;
        String[] tokens = new String[baseTokens + 5];
        tokens[0] = "a";
        tokens[1] = "b";
        tokens[2] = "ab";
        for (int i = 3; i < baseTokens; i++) {
            tokens[i] = "token_" + i;
        }
        tokens[baseTokens] = "<|begin_of_text|>";
        tokens[baseTokens + 1] = "<|start_header_id|>";
        tokens[baseTokens + 2] = "<|end_header_id|>";
        tokens[baseTokens + 3] = "<|eot_id|>";
        tokens[baseTokens + 4] = "<|end_of_text|>";
        metadata.put("tokenizer.ggml.tokens", tokens);
        metadata.put("tokenizer.ggml.merges", new String[]{"a b"});

        metadata.put("llama.embedding_length", 4);
        metadata.put("llama.feed_forward_length", 8);
        metadata.put("llama.block_count", 1);
        metadata.put("llama.attention.head_count", 1);
        metadata.put("llama.attention.head_count_kv", 1);
        metadata.put("llama.context_length", 16);
        metadata.put("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        metadata.put("llama.rope.freq_base", 10000f);

        Llama model = ModelLoader.buildModelFromMetadata(metadata, 8);

        assertNotNull(model.tokenizer());
        assertEquals(8, model.configuration().contextLength);
        assertEquals(baseTokens + 5, model.configuration().vocabularySize);
    }
}
