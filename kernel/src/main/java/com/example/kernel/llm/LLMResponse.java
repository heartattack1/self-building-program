package com.example.kernel.llm;

import java.util.Map;

public record LLMResponse(String text, Map<String, Object> metrics) {
}
