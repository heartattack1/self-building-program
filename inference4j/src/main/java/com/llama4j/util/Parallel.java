package com.llama4j.util;

import java.util.function.IntConsumer;
import java.util.function.LongConsumer;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * Executes simple parallel for loops with sane fallbacks for single-iteration cases.
 */
public final class Parallel {
    private Parallel() {
    }

    /**
     * Runs an {@link IntConsumer} in parallel for the given integer range.
     *
     * @param startInclusive start of the range (inclusive)
     * @param endExclusive end of the range (exclusive)
     * @param action action to apply to each index
     */
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    /**
     * Runs a {@link LongConsumer} in parallel for the given long range.
     *
     * @param startInclusive start of the range (inclusive)
     * @param endExclusive end of the range (exclusive)
     * @param action action to apply to each index
     */
    public static void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}
