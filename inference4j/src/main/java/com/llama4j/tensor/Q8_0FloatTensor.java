package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q8_0} format.
 */
public final class Q8_0FloatTensor extends FloatTensor {
    private final int size;
    private final java.lang.foreign.MemorySegment memorySegment;

    /**
     * Creates a Q8_0 tensor.
     *
     * @param size number of elements
     * @param memorySegment tensor data segment
     */
    public Q8_0FloatTensor(int size, java.lang.foreign.MemorySegment memorySegment) {
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
        return GGMLType.Q8_0;
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
        int blockIndex = index / GGMLType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        byte quant = readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + (index % GGMLType.Q8_0.getBlockSize()));
        return quant * scale;
    }

    /**
     * Computes dot product with potential vectorized path.
     *
     * @param thisOffset offset in this tensor
     * @param that other tensor
     * @param thatOffset offset in other tensor
     * @param size number of elements
     * @return dot product
     */
    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() >= 512) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    /**
     * Vectorized dot product specialized for Q8_0 weights.
     *
     * @param thiz quantized tensor
     * @param thisOffset offset in quantized tensor
     * @param that dense tensor
     * @param thatOffset offset in dense tensor
     * @param size number of elements
     * @return dot product
     */
    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = size / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment,
                    blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            var wBytesVector = wBytes.convertShape(VectorOperators.B2I, I_SPECIES, 0).reinterpretAsFloats();
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(wBytesVector.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length())
                            .mul(wBytesVector.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(wBytesVector.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length())
                            .mul(wBytesVector.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length())
                            .mul(wBytesVector.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length())
                            .mul(wBytesVector.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 2) * F_SPECIES.length())
                                .mul(wBytesVector.castShape(F_SPECIES, i * 2));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 2 + 1) * F_SPECIES.length())
                                .mul(wBytesVector.castShape(F_SPECIES, i * 2 + 1));
                        val = sum0.add(sum1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}
