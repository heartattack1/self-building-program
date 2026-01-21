package com.llama4j;

import com.llama4j.model.Llama;
import com.llama4j.sampling.Sampler;
import com.llama4j.tensor.ArrayFloatTensor;
import com.llama4j.tensor.FloatTensor;
import com.llama4j.tensor.RoPE;
import com.llama4j.tokenizer.Tokenizer;
import com.llama4j.tokenizer.Vocabulary;
import com.llama4j.util.Pair;

import org.junit.jupiter.api.Test;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for Llama generation.
 */
class LlamaTest {
    /**
     * Confirms generation stops when a stop token is produced.
     */
    @Test
    void generateTokensStopsOnStopToken() {
        String[] tokens = {"a", "b", "c", "<|begin_of_text|>"};
        Vocabulary vocabulary = new Vocabulary(tokens, null);
        Map<String, Integer> specialTokens = Map.of("<|begin_of_text|>", 3);
        Tokenizer tokenizer = new Tokenizer(vocabulary, List.of(), ".", specialTokens);

        Llama.Configuration config = new Llama.Configuration(4, 4, 1, 1, 1, tokens.length, 4, 1e-5f, 10000f);
        Llama.Weights weights = buildZeroWeights(config);
        Llama model = new Llama(config, tokenizer, weights);
        Llama.State state = model.createNewState(1);

        List<Integer> generated = Llama.generateTokens(model, state, 0, List.of(0), Set.of(0), 4, Sampler.ARGMAX, false, null);

        assertEquals(List.of(0), generated);
    }

    /**
     * Builds zero-initialized weights for a tiny test model.
     *
     * @param config model configuration
     * @return zero weights
     */
    private static Llama.Weights buildZeroWeights(Llama.Configuration config) {
        FloatTensor tokenEmbeddingTable = ArrayFloatTensor.allocate(config.vocabularySize, config.dim);
        FloatBuffer[] rmsAttWeight = {FloatBuffer.wrap(new float[config.dim])};
        FloatTensor[] wq = {ArrayFloatTensor.allocate(config.dim, config.dim)};
        FloatTensor[] wk = {ArrayFloatTensor.allocate(config.dim, config.dim)};
        FloatTensor[] wv = {ArrayFloatTensor.allocate(config.dim, config.dim)};
        FloatTensor[] wo = {ArrayFloatTensor.allocate(config.dim, config.dim)};
        FloatBuffer[] rmsFfnWeight = {FloatBuffer.wrap(new float[config.dim])};
        FloatTensor[] w1 = {ArrayFloatTensor.allocate(config.hiddenDim, config.dim)};
        FloatTensor[] w2 = {ArrayFloatTensor.allocate(config.dim, config.hiddenDim)};
        FloatTensor[] w3 = {ArrayFloatTensor.allocate(config.hiddenDim, config.dim)};
        FloatBuffer rmsFinalWeight = FloatBuffer.wrap(new float[config.dim]);

        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                false, 1f, 1f, 1f, config.contextLength);
        FloatBuffer ropeReal = FloatBuffer.wrap(ropeFreqs.first());
        FloatBuffer ropeImag = FloatBuffer.wrap(ropeFreqs.second());
        FloatTensor wcls = ArrayFloatTensor.allocate(config.vocabularySize, config.dim);

        return new Llama.Weights(tokenEmbeddingTable, rmsAttWeight, wq, wk, wv, wo, rmsFfnWeight,
                w1, w2, w3, rmsFinalWeight, ropeReal, ropeImag, wcls);
    }
}
