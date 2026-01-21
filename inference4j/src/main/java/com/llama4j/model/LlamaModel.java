package com.llama4j.model;

import com.llama4j.config.ModelConfiguration;
import com.llama4j.sampling.Sampler;
import com.llama4j.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

/**
 * Llama model adapter that implements the generic {@link Model} interface.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * LlamaModel model = new LlamaModel();
 * model.loadModel(Path.of("/path/to/llama.gguf"));
 * List<Integer> response = model.generateResponse(promptTokens, Sampler.ARGMAX, 128);
 * }</pre>
 */
public final class LlamaModel implements Model {
    /**
     * Loader strategy for Llama models.
     */
    @FunctionalInterface
    public interface Loader {
        Llama load(Path modelPath, int contextLength, boolean loadWeights) throws IOException;
    }

    /**
     * Inference strategy for Llama models.
     */
    @FunctionalInterface
    public interface InferenceEngine {
        List<Integer> generate(Llama model, Llama.State state, int startPosition, List<Integer> promptTokens,
                               Set<Integer> stopTokens, int maxTokens, Sampler sampler);
    }

    private final Loader loader;
    private final InferenceEngine inferenceEngine;
    private final int batchSize;
    private Llama llama;

    /**
     * Creates a Llama model adapter with default loading and inference logic.
     */
    public LlamaModel() {
        this(ModelLoader::loadModel,
                (model, state, start, promptTokens, stopTokens, maxTokens, sampler) ->
                        Llama.generateTokens(model, state, start, promptTokens, stopTokens, maxTokens, sampler, false, null),
                1);
    }

    /**
     * Creates a Llama model adapter with injected dependencies.
     *
     * @param loader loader implementation
     * @param inferenceEngine inference engine
     * @param batchSize batch size for prompt evaluation
     */
    public LlamaModel(Loader loader, InferenceEngine inferenceEngine, int batchSize) {
        this.loader = loader;
        this.inferenceEngine = inferenceEngine;
        this.batchSize = batchSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void loadModel(Path modelPath) throws IOException {
        this.llama = loader.load(modelPath, -1, true);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Integer> generateResponse(List<Integer> promptTokens, Sampler sampler, int maxTokens) {
        ensureLoaded();
        Llama.State state = llama.createNewState(batchSize);
        return inferenceEngine.generate(llama, state, 0, promptTokens, Set.of(), maxTokens, sampler);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tokenizer tokenizer() {
        ensureLoaded();
        return llama.tokenizer();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ModelConfiguration configuration() {
        ensureLoaded();
        return llama.configuration();
    }

    private void ensureLoaded() {
        if (llama == null) {
            throw new IllegalStateException("Model has not been loaded");
        }
    }
}
