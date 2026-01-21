package com.llama4j.sampling;

import com.llama4j.tensor.FloatTensor;

/**
 * Strategy for selecting a token index from logits.
 */
@FunctionalInterface
public interface Sampler {
    /**
     * Picks a token index from the provided logits.
     *
     * @param logits logits or probabilities to sample from
     * @return selected token index
     */
    int sampleToken(FloatTensor logits);

    /**
     * Greedy argmax sampling strategy.
     */
    Sampler ARGMAX = FloatTensor::argmax;
}
