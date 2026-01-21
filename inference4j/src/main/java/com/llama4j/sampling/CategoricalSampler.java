package com.llama4j.sampling;

import com.llama4j.tensor.FloatTensor;

import java.util.random.RandomGenerator;

/**
 * Samples a token according to the categorical distribution defined by probabilities.
 *
 * @param rng random number generator
 */
public record CategoricalSampler(RandomGenerator rng) implements Sampler {

    /**
     * Samples a token by traversing the cumulative distribution.
     *
     * @param logits probability distribution (assumes sum to 1)
     * @return sampled token index
     */
    @Override
    public int sampleToken(FloatTensor logits) {
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.size() - 1;
    }
}
