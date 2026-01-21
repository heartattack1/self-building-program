package com.example.kernel.llm;

public interface LLMAdapter {
    LLMResponse generate(LLMRequest request);
}
