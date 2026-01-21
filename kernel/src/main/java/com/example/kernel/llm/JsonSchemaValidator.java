package com.example.kernel.llm;

import com.example.kernel.GeneratedSourceBundle;
import com.example.kernel.Plan;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class JsonSchemaValidator {
    public Plan parsePlan(JsonNode node) {
        if (node == null || !node.isObject()) {
            throw new IllegalArgumentException("Plan JSON must be an object");
        }
        JsonNode versionIdNode = node.get("versionId");
        JsonNode implNode = node.get("implMainClass");
        JsonNode filesNode = node.get("files");
        JsonNode invariantsNode = node.get("invariants");
        JsonNode notesNode = node.get("notes");

        if (!isText(versionIdNode) || !isText(implNode)) {
            throw new IllegalArgumentException("Plan JSON missing versionId or implMainClass");
        }
        if (filesNode == null || !filesNode.isArray()) {
            throw new IllegalArgumentException("Plan JSON missing files array");
        }
        if (invariantsNode == null || !invariantsNode.isArray()) {
            throw new IllegalArgumentException("Plan JSON missing invariants array");
        }
        if (notesNode == null || !notesNode.isTextual()) {
            throw new IllegalArgumentException("Plan JSON missing notes");
        }

        List<String> tasks = new ArrayList<>();
        for (JsonNode fileNode : filesNode) {
            JsonNode fqcnNode = fileNode.get("fqcn");
            JsonNode roleNode = fileNode.get("role");
            if (!isText(fqcnNode) || !isText(roleNode)) {
                throw new IllegalArgumentException("Plan files entries must include fqcn and role");
            }
            tasks.add("Generate " + fqcnNode.asText() + " (" + roleNode.asText() + ")");
        }
        for (JsonNode invariantNode : invariantsNode) {
            if (invariantNode.isTextual()) {
                tasks.add("Invariant: " + invariantNode.asText());
            }
        }
        if (!notesNode.asText().isBlank()) {
            tasks.add("Notes: " + notesNode.asText());
        }

        return new Plan(versionIdNode.asText(), implNode.asText(), tasks);
    }

    public CodeGenResult parseCodeGen(JsonNode node) {
        if (node == null || !node.isObject()) {
            throw new IllegalArgumentException("CodeGen JSON must be an object");
        }
        JsonNode versionIdNode = node.get("versionId");
        JsonNode filesNode = node.get("files");
        JsonNode notesNode = node.get("notes");
        if (!isText(versionIdNode)) {
            throw new IllegalArgumentException("CodeGen JSON missing versionId");
        }
        if (filesNode == null || !filesNode.isObject()) {
            throw new IllegalArgumentException("CodeGen JSON missing files object");
        }
        if (notesNode == null || !notesNode.isTextual()) {
            throw new IllegalArgumentException("CodeGen JSON missing notes");
        }
        Map<String, String> files = new HashMap<>();
        Iterator<Map.Entry<String, JsonNode>> fields = filesNode.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> entry = fields.next();
            if (!entry.getValue().isTextual()) {
                throw new IllegalArgumentException("CodeGen file sources must be strings");
            }
            files.put(entry.getKey(), entry.getValue().asText());
        }
        return new CodeGenResult(versionIdNode.asText(), new GeneratedSourceBundle(files), notesNode.asText());
    }

    private boolean isText(JsonNode node) {
        return node != null && node.isTextual() && !node.asText().isBlank();
    }

    public record CodeGenResult(String versionId, GeneratedSourceBundle sources, String notes) {
    }
}
