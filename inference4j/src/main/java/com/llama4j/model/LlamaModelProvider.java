package com.llama4j.model;

/**
 * Service provider for Llama models.
 */
public class LlamaModelProvider implements ModelProvider {
    @Override
    public String modelName() {
        return "llama";
    }

    @Override
    public Model createModel() {
        return new LlamaModel();
    }
}
