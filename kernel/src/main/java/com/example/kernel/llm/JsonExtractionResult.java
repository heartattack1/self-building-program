package com.example.kernel.llm;

import com.fasterxml.jackson.databind.JsonNode;

public record JsonExtractionResult(
        JsonNode node,
        String extracted,
        String error
) {
    public boolean success() {
        return node != null;
    }
}
