package com.llama4j.sampling;

import com.llama4j.tensor.FloatTensor;

import java.util.Comparator;
import java.util.random.RandomGenerator;

/**
 * Top-p (nucleus) sampling strategy.
 */
public final class ToppSampler implements Sampler {
    private final int[] indices;
    private final float topp;
    private final RandomGenerator rng;

    /**
     * Creates a top-p sampler.
     *
     * @param maxNumberOfElements max size for internal index buffer
     * @param topp cumulative probability threshold
     * @param rng random number generator
     */
    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    /**
     * Swaps two array elements.
     *
     * @param array array to mutate
     * @param from source index
     * @param to destination index
     */
    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    /**
     * Restores heap ordering from the provided index.
     *
     * @param array heap array
     * @param from root index
     * @param n heap size
     * @param comparator comparator for ordering
     */
    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from;
        int next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    /**
     * Samples from the smallest set of tokens whose cumulative probability exceeds {@code topp}.
     *
     * @param logits probability distribution (assumes sum to 1)
     * @return sampled token index
     */
    @Override
    public int sampleToken(FloatTensor logits) {
        Comparator<Integer> comparator = Comparator.comparingDouble(logits::getFloat).reversed();

        int n = logits.size();
        int head = 0;
        int tail = n - 1;
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break;
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex];
    }
}
