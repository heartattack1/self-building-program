package com.llama4j.model;

import com.llama4j.sampling.Sampler;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Gpt2ModelTest {
    @Test
    void generatesDeterministicTokens() throws Exception {
        Gpt2Model model = new Gpt2Model();
        model.loadModel(Path.of("gpt2.bin"));

        List<Integer> response = model.generateResponse(List.of(1, 2), Sampler.ARGMAX, 2);

        assertEquals(List.of(3, 3), response);
    }
}
