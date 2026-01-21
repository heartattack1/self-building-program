package com.llama4j.model;

/**
 * Service provider for GPT-2 models.
 */
public class Gpt2ModelProvider implements ModelProvider {
    @Override
    public String modelName() {
        return "gpt2";
    }

    @Override
    public Model createModel() {
        return new Gpt2Model();
    }
}
