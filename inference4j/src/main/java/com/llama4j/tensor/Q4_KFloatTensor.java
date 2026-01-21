package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_K} format.
 */
public final class Q4_KFloatTensor extends FloatTensor {
    private static final int SUB_BLOCK_COUNT = 8;
    private static final int SUB_BLOCK_SIZE = GGMLType.QK_K / SUB_BLOCK_COUNT;
    private static final int SCALES_LENGTH = (GGMLType.QK_K / 16) / 8 * 6;
    private static final int HEADER_BYTES = 2 * GGMLType.FLOAT16_BYTES;
    private static final int SCALES_OFFSET = HEADER_BYTES;
    private static final int QS_OFFSET = HEADER_BYTES + SCALES_LENGTH;

    private final int size;
    private final java.lang.foreign.MemorySegment memorySegment;

    /**
     * Creates a Q4_K tensor.
     *
     * @param size number of elements
     * @param memorySegment tensor data segment
     */
    public Q4_KFloatTensor(int size, java.lang.foreign.MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    /**
     * Returns the number of elements.
     *
     * @return element count
     */
    @Override
    public int size() {
        return size;
    }

    /**
     * Unsupported for quantized tensor.
     *
     * @param index element index
     * @param value value to set
     */
    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    /**
     * Unsupported for quantized tensor.
     *
     * @param species vector species
     * @param index offset index
     * @return float vector
     */
    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    /**
     * Returns the GGML type.
     *
     * @return GGML type
     */
    @Override
    public GGMLType type() {
        return GGMLType.Q4_K;
    }

    /**
     * Reads a dequantized float value.
     *
     * @param index element index
     * @return dequantized value
     */
    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_K.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_K.getTypeSize();
        float blockScale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        float blockMin = Float.float16ToFloat(readShort(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES));

        int blockElement = index % GGMLType.Q4_K.getBlockSize();
        int subBlock = blockElement / SUB_BLOCK_SIZE;

        int scale = readScale(memorySegment, blockOffset + SCALES_OFFSET, subBlock);
        int min = readScale(memorySegment, blockOffset + SCALES_OFFSET, subBlock + SUB_BLOCK_COUNT);
        float scaleValue = blockScale * scale;
        float minValue = blockMin * min;

        int quantIndex = blockElement;
        int quantByteIndex = quantIndex / 2;
        byte quantByte = readByte(memorySegment, blockOffset + QS_OFFSET + quantByteIndex);
        int quant = (quantIndex % 2 == 0) ? (quantByte & 0x0F) : ((quantByte >>> 4) & 0x0F);

        return scaleValue * quant + minValue;
    }

    /**
     * Computes dot product.
     *
     * @param thisOffset offset in this tensor
     * @param that other tensor
     * @param thatOffset offset in other tensor
     * @param size number of elements
     * @return dot product
     */
    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static int readScale(java.lang.foreign.MemorySegment memorySegment, long scalesOffset, int index) {
        int bitOffset = index * 6;
        int byteIndex = bitOffset / 8;
        int bitIndex = bitOffset % 8;
        int b0 = readByte(memorySegment, scalesOffset + byteIndex) & 0xFF;
        if (bitIndex <= 2) {
            return (b0 >> bitIndex) & 0x3F;
        }
        int b1 = readByte(memorySegment, scalesOffset + byteIndex + 1) & 0xFF;
        return ((b0 >> bitIndex) | (b1 << (8 - bitIndex))) & 0x3F;
    }
}
