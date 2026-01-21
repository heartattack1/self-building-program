package com.llama4j.util;

import java.util.concurrent.TimeUnit;

/**
 * Utility for timing operations with automatic logging.
 */
public interface Timer extends AutoCloseable {
    /**
     * Stops the timer and emits the elapsed time.
     */
    @Override
    void close();

    /**
     * Creates a timer that logs elapsed time in milliseconds.
     *
     * @param label label to prepend to the log message
     * @return a timer instance
     */
    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    /**
     * Creates a timer that logs elapsed time in the provided time unit.
     *
     * @param label label to prepend to the log message
     * @param timeUnit time unit to log
     * @return a timer instance
     */
    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}
