package com.llama4j.model;

import com.llama4j.config.Gpt2Configuration;
import com.llama4j.config.ModelConfiguration;
import com.llama4j.sampling.Sampler;
import com.llama4j.tensor.ArrayFloatTensor;
import com.llama4j.tensor.FloatTensor;
import com.llama4j.tokenizer.Tokenizer;
import com.llama4j.tokenizer.TokenizerConfig;
import com.llama4j.tokenizer.TokenizerFactory;
import com.llama4j.tokenizer.Vocabulary;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Minimal GPT-2 model adapter that implements the {@link Model} interface.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * Model model = ModelFactory.createModel("gpt2");
 * model.loadModel(Path.of("/path/to/gpt2.bin"));
 * List<Integer> response = model.generateResponse(promptTokens, Sampler.ARGMAX, 64);
 * }</pre>
 */
public final class Gpt2Model implements Model {
    private static final String MODEL_NAME = "gpt2";

    private ModelConfiguration configuration;
    private Tokenizer tokenizer;
    private boolean loaded;

    /**
     * {@inheritDoc}
     */
    @Override
    public void loadModel(Path modelPath) throws IOException {
        String[] tokens = {"<unk>", "hello", "world", "<eos>"};
        Vocabulary vocabulary = new Vocabulary(tokens, null);
        TokenizerConfig config = TokenizerConfig.builder()
                .vocabulary(vocabulary)
                .specialTokens(Map.of("<eos>", 3))
                .unknownToken("<unk>")
                .build();
        this.tokenizer = TokenizerFactory.createTokenizer(MODEL_NAME, config);
        this.configuration = new Gpt2Configuration(vocabulary.size(), 128, 12, 12, 768);
        this.loaded = true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Integer> generateResponse(List<Integer> promptTokens, Sampler sampler, int maxTokens) {
        ensureLoaded();
        int capped = Math.max(0, maxTokens);
        List<Integer> output = new ArrayList<>(capped);
        FloatTensor logits = ArrayFloatTensor.allocate(configuration.vocabularySize());
        for (int i = 0; i < logits.size(); i++) {
            logits.setFloat(i, i);
        }
        for (int i = 0; i < capped; i++) {
            output.add(sampler.sampleToken(logits));
        }
        return output;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tokenizer tokenizer() {
        ensureLoaded();
        return tokenizer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ModelConfiguration configuration() {
        ensureLoaded();
        return configuration;
    }

    private void ensureLoaded() {
        if (!loaded) {
            throw new IllegalStateException("Model has not been loaded");
        }
    }
}
