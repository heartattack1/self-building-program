package com.llama4j.model;

import com.llama4j.sampling.Sampler;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BertModelTest {
    @Test
    void generatesDeterministicTokens() throws Exception {
        BertModel model = new BertModel();
        model.loadModel(Path.of("bert.bin"));

        List<Integer> response = model.generateResponse(List.of(1, 2), Sampler.ARGMAX, 1);

        assertEquals(List.of(0), response);
    }
}
