package com.llama4j.model;

/**
 * Service provider for BERT models.
 */
public class BertModelProvider implements ModelProvider {
    @Override
    public String modelName() {
        return "bert";
    }

    @Override
    public Model createModel() {
        return new BertModel();
    }
}
