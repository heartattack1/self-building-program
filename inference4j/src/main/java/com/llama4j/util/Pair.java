package com.llama4j.util;

/**
 * Simple immutable pair for transporting two related values.
 *
 * @param first first value
 * @param second second value
 */
public record Pair<First, Second>(First first, Second second) {
}
