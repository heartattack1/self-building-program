package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q6_K} format.
 */
public final class Q6_KFloatTensor extends FloatTensor {
    private static final int SUB_BLOCK_SIZE = 16;
    private static final int HEADER_BYTES = GGMLType.FLOAT16_BYTES;
    private static final int QL_LENGTH = GGMLType.QK_K / 2;
    private static final int QH_LENGTH = GGMLType.QK_K / 4;
    private static final int SCALES_LENGTH = GGMLType.QK_K / SUB_BLOCK_SIZE;
    private static final int QL_OFFSET = HEADER_BYTES;
    private static final int QH_OFFSET = QL_OFFSET + QL_LENGTH;
    private static final int SCALES_OFFSET = QH_OFFSET + QH_LENGTH;

    private final int size;
    private final java.lang.foreign.MemorySegment memorySegment;

    /**
     * Creates a Q6_K tensor.
     *
     * @param size number of elements
     * @param memorySegment tensor data segment
     */
    public Q6_KFloatTensor(int size, java.lang.foreign.MemorySegment memorySegment) {
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
        return GGMLType.Q6_K;
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
        int blockIndex = index / GGMLType.Q6_K.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q6_K.getTypeSize();
        float blockScale = Float.float16ToFloat(readShort(memorySegment, blockOffset));

        int blockElement = index % GGMLType.Q6_K.getBlockSize();
        int scale = readByte(memorySegment, blockOffset + SCALES_OFFSET + (blockElement / SUB_BLOCK_SIZE));

        int qlIndex = blockElement / 2;
        int qlShift = (blockElement % 2) * 4;
        int ql = (readByte(memorySegment, blockOffset + QL_OFFSET + qlIndex) >> qlShift) & 0x0F;

        int qhIndex = blockElement / 4;
        int qhShift = (blockElement % 4) * 2;
        int qh = (readByte(memorySegment, blockOffset + QH_OFFSET + qhIndex) >> qhShift) & 0x03;

        int quant = ql | (qh << 4);
        return blockScale * scale * (quant - 32);
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
}
