package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * Dense float tensor backed by a Java array.
 */
public final class ArrayFloatTensor extends FloatTensor {
    private final float[] values;

    /**
     * Creates a tensor backed by the provided values.
     *
     * @param values backing array
     */
    public ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    /**
     * Allocates a new dense tensor with the provided dimensions.
     *
     * @param dims tensor dimensions
     * @return allocated tensor
     */
    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    /**
     * Returns the number of elements.
     *
     * @return element count
     */
    @Override
    public int size() {
        return values.length;
    }

    /**
     * Gets the float value at the given index.
     *
     * @param index element index
     * @return value
     */
    @Override
    public float getFloat(int index) {
        return values[index];
    }

    /**
     * Sets the float value at the given index.
     *
     * @param index element index
     * @param value value to set
     */
    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    /**
     * Returns the GGML type.
     *
     * @return GGML type
     */
    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    /**
     * Fills a slice with a constant value.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @param value value to set
     * @return this tensor
     */
    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    /**
     * Loads a float vector from the backing array.
     *
     * @param species vector species
     * @param index offset
     * @return vector of floats
     */
    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
}
