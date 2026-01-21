package com.llama4j.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class ModelFactoryTest {
    @Test
    void createsRegisteredModels() {
        assertInstanceOf(LlamaModel.class, ModelFactory.createModel("llama"));
        assertInstanceOf(Gpt2Model.class, ModelFactory.createModel("gpt2"));
        assertInstanceOf(BertModel.class, ModelFactory.createModel("bert"));
    }
}
