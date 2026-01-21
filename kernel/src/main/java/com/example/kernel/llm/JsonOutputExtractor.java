package com.example.kernel.llm;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayDeque;
import java.util.Deque;

public class JsonOutputExtractor {
    private final ObjectMapper mapper;

    public JsonOutputExtractor(ObjectMapper mapper) {
        this.mapper = mapper;
    }

    public JsonExtractionResult extract(String output) {
        if (output == null || output.isBlank()) {
            return new JsonExtractionResult(null, null, "Empty output");
        }
        String trimmed = output.trim();
        JsonNode node = parse(trimmed);
        if (node != null) {
            return new JsonExtractionResult(node, trimmed, null);
        }

        String extracted = extractFirstJson(trimmed);
        if (extracted == null) {
            return new JsonExtractionResult(null, null, "No JSON object or array found");
        }
        node = parse(extracted);
        if (node != null) {
            return new JsonExtractionResult(node, extracted, null);
        }
        return new JsonExtractionResult(null, extracted, "Failed to parse extracted JSON");
    }

    private JsonNode parse(String text) {
        try {
            return mapper.readTree(text);
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    private String extractFirstJson(String text) {
        int length = text.length();
        for (int start = 0; start < length; start++) {
            char ch = text.charAt(start);
            if (ch == '{' || ch == '[') {
                int end = findMatchingBracket(text, start);
                if (end > start) {
                    return text.substring(start, end + 1);
                }
            }
        }
        return null;
    }

    private int findMatchingBracket(String text, int start) {
        Deque<Character> stack = new ArrayDeque<>();
        boolean inString = false;
        boolean escape = false;
        for (int i = start; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (inString) {
                if (escape) {
                    escape = false;
                } else if (ch == '\\\\') {
                    escape = true;
                } else if (ch == '"') {
                    inString = false;
                }
                continue;
            }
            if (ch == '"') {
                inString = true;
                continue;
            }
            if (ch == '{' || ch == '[') {
                stack.push(ch);
            } else if (ch == '}' || ch == ']') {
                if (stack.isEmpty()) {
                    return -1;
                }
                char open = stack.pop();
                if ((open == '{' && ch != '}') || (open == '[' && ch != ']')) {
                    return -1;
                }
                if (stack.isEmpty()) {
                    return i;
                }
            }
        }
        return -1;
    }
}
