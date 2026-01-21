package com.example.kernel.llm;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Inference4jLLMAdapterTest {
    @Test
    void enforcesMaxOutputCharsAndReportsMetrics() {
        InferenceBackend backend = request -> new InferenceResult("{\"status\":\"ok\"}", 5);
        Inference4jLLMAdapter adapter = new Inference4jLLMAdapter(backend, 1_000, 5, 0.9);
        LLMResponse response = adapter.generate(new LLMRequest(
                "plan",
                "specHash",
                "prompt",
                16,
                0.2,
                123,
                Map.of()
        ));

        assertEquals("{\"sta", response.text());
        assertEquals(5, response.metrics().get("generatedTokens"));
        assertTrue((Boolean) response.metrics().get("truncated"));
    }
}
