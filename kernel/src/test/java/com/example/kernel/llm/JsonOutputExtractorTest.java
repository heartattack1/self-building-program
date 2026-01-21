package com.example.kernel.llm;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonOutputExtractorTest {
    private final JsonOutputExtractor extractor = new JsonOutputExtractor(new ObjectMapper());

    @Test
    void extractsEntireJsonWhenValid() {
        String json = "{\"versionId\":\"v1\",\"implMainClass\":\"com.example.impl.v1.GeneratedService\",\"files\":[],\"invariants\":[],\"notes\":\"ok\"}";
        JsonExtractionResult result = extractor.extract(json);
        assertTrue(result.success());
        assertEquals(json, result.extracted());
        assertNotNull(result.node());
        assertEquals("v1", result.node().get("versionId").asText());
    }

    @Test
    void extractsJsonFromNoisyOutput() {
        String output = "noise before {\"versionId\":\"v1\",\"files\":{},\"notes\":\"ok\"} trailing";
        JsonExtractionResult result = extractor.extract(output);
        assertTrue(result.success());
        assertNotNull(result.extracted());
        assertTrue(result.extracted().startsWith("{\"versionId\":\"v1\""));
        assertNotNull(result.node());
        assertEquals("v1", result.node().get("versionId").asText());
    }
}
