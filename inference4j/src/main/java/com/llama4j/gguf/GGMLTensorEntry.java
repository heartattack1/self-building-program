package com.llama4j.gguf;

import java.lang.foreign.MemorySegment;

/**
 * Represents a tensor entry in a GGUF file, including its metadata and memory segment.
 *
 * @param mappedFile memory-mapped GGUF file segment
 * @param name tensor name
 * @param ggmlType GGML tensor type
 * @param shape tensor shape
 * @param memorySegment memory segment with the tensor data
 */
public record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                              MemorySegment memorySegment) {
}
