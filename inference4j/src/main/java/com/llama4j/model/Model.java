package com.llama4j.model;

import com.llama4j.config.ModelConfiguration;
import com.llama4j.sampling.Sampler;
import com.llama4j.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Common interface for language models.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * Model model = ModelFactory.createModel("llama");
 * model.loadModel(Path.of("/path/to/model.gguf"));
 * List<Integer> response = model.generateResponse(promptTokens, Sampler.ARGMAX, 128);
 * }</pre>
 */
public interface Model {
    /**
     * Loads the model from the given path.
     *
     * @param modelPath model file path
     * @throws IOException when loading fails
     */
    void loadModel(Path modelPath) throws IOException;

    /**
     * Generates a response from the prompt tokens.
     *
     * @param promptTokens input prompt tokens
     * @param sampler sampling strategy
     * @param maxTokens maximum number of generated tokens
     * @return generated tokens
     */
    List<Integer> generateResponse(List<Integer> promptTokens, Sampler sampler, int maxTokens);

    /**
     * Returns the model tokenizer.
     *
     * @return tokenizer
     */
    Tokenizer tokenizer();

    /**
     * Returns the model configuration.
     *
     * @return configuration
     */
    ModelConfiguration configuration();
}
