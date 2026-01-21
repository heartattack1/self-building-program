package com.example.kernel.llm;

import com.example.kernel.GeneratedSourceBundle;
import com.example.kernel.Plan;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonSchemaValidatorTest {
    private final ObjectMapper mapper = new ObjectMapper();
    private final JsonSchemaValidator validator = new JsonSchemaValidator();

    @Test
    void parsesPlanPayload() throws Exception {
        String json = """
                {
                  "versionId": "v1-abc",
                  "implMainClass": "com.example.impl.v1.GeneratedService",
                  "files": [{"fqcn":"com.example.impl.v1.GeneratedService","role":"impl"}],
                  "invariants": ["Always return OK"],
                  "notes": "ok"
                }
                """;
        JsonNode node = mapper.readTree(json);
        Plan plan = validator.parsePlan(node);
        assertEquals("v1-abc", plan.versionId());
        assertEquals("com.example.impl.v1.GeneratedService", plan.implClassName());
        assertTrue(plan.tasks().stream().anyMatch(task -> task.contains("Invariant: Always return OK")));
    }

    @Test
    void parsesCodeGenPayload() throws Exception {
        String json = """
                {
                  "versionId": "v1-abc",
                  "files": {
                    "com.example.impl.v1.GeneratedService": "class GeneratedService {}"
                  },
                  "notes": "ok"
                }
                """;
        JsonNode node = mapper.readTree(json);
        JsonSchemaValidator.CodeGenResult result = validator.parseCodeGen(node);
        GeneratedSourceBundle bundle = result.sources();
        assertEquals("v1-abc", result.versionId());
        assertEquals(Map.of("com.example.impl.v1.GeneratedService", "class GeneratedService {}"), bundle.sources());
    }
}
