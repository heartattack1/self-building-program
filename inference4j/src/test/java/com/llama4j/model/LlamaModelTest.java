package com.llama4j.model;

import com.llama4j.sampling.Sampler;
import com.llama4j.tokenizer.SimpleTokenizer;
import com.llama4j.tokenizer.Vocabulary;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LlamaModelTest {
    @Test
    void generateResponseUsesInjectedInferenceEngine() throws Exception {
        AtomicReference<Path> loadedPath = new AtomicReference<>();
        LlamaModel.Loader loader = (path, contextLength, loadWeights) -> {
            loadedPath.set(path);
            String[] tokens = {"<|begin_of_text|>", "hello"};
            Vocabulary vocabulary = new Vocabulary(tokens, null);
            SimpleTokenizer tokenizer = new SimpleTokenizer(vocabulary, Map.of("<|begin_of_text|>", 0), "<unk>");
            Llama.Configuration configuration = new Llama.Configuration(4, 8, 1, 1, 1, vocabulary.size(), 8, 1e-5f, 10000f);
            return new Llama(configuration, tokenizer, null);
        };
        List<Integer> expected = List.of(1, 1, 1);
        LlamaModel.InferenceEngine inferenceEngine = (model, state, startPosition, promptTokens, stopTokens, maxTokens, sampler) -> expected;

        LlamaModel model = new LlamaModel(loader, inferenceEngine, 1);
        model.loadModel(Path.of("fake.gguf"));

        List<Integer> response = model.generateResponse(List.of(0), Sampler.ARGMAX, 3);

        assertEquals(Path.of("fake.gguf"), loadedPath.get());
        assertEquals(expected, response);
    }

    @Test
    void generateResponseFailsWhenNotLoaded() {
        LlamaModel model = new LlamaModel();

        assertThrows(IllegalStateException.class, () -> model.generateResponse(List.of(0), Sampler.ARGMAX, 1));
    }
}
