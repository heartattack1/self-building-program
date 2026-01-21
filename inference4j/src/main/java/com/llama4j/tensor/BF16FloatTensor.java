package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;

/**
 * {@link FloatTensor} backed by BF16 values.
 */
public final class BF16FloatTensor extends FloatTensor {
    private final int size;
    private final java.lang.foreign.MemorySegment memorySegment;

    /**
     * Creates a BF16 tensor.
     *
     * @param size number of elements
     * @param memorySegment tensor data segment
     */
    public BF16FloatTensor(int size, java.lang.foreign.MemorySegment memorySegment) {
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
     * Unsupported for BF16 tensor.
     *
     * @param index element index
     * @param value value to set
     */
    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    /**
     * Unsupported for BF16 tensor.
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
        return GGMLType.BF16;
    }

    /**
     * Reads a value as float.
     *
     * @param index element index
     * @return float value
     */
    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return bfloat16ToFloat(readShort(memorySegment, (long) index * GGMLType.FLOAT16_BYTES));
    }

    /**
     * Converts BF16 to float.
     *
     * @param bfloat16 BF16 value
     * @return float value
     */
    private float bfloat16ToFloat(short bfloat16) {
        return Float.intBitsToFloat(bfloat16 << 16);
    }

    /**
     * Computes dot product using vectorized BF16 path when possible.
     *
     * @param thisOffset offset in this tensor
     * @param that other tensor
     * @param thatOffset offset in other tensor
     * @param size number of elements
     * @return dot product
     */
    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    /**
     * Vectorized dot product specialized for BF16 values.
     *
     * @param thiz BF16 tensor
     * @param thisOffset offset in BF16 tensor
     * @param that dense tensor
     * @param thatOffset offset in dense tensor
     * @param size number of elements
     * @return dot product
     */
    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment,
                    (thisOffset + i) * (long) GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0)
                    .lanewise(VectorOperators.LSHL, 16)
                    .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }
}
