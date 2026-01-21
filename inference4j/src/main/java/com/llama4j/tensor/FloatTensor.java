package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;
import com.llama4j.util.Parallel;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;
import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.util.Arrays;

/**
 * Base class for float tensor implementations with optional Vector API acceleration.
 */
public abstract class FloatTensor {
    /**
     * Configurable vector width. Defaults to preferred width when 0.
     */
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());

    /**
     * Flag to enable Vector API usage.
     */
    public static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    /**
     * Unsafe instance for fast memory operations.
     */
    protected static final Unsafe UNSAFE;

    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    /**
     * Reads a little-endian short from a memory segment.
     *
     * @param memorySegment memory segment
     * @param offset offset in bytes
     * @return short value
     */
    static short readShort(java.lang.foreign.MemorySegment memorySegment, long offset) {
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    /**
     * Reads a byte from a memory segment.
     *
     * @param memorySegment memory segment
     * @param offset offset in bytes
     * @return byte value
     */
    static byte readByte(java.lang.foreign.MemorySegment memorySegment, long offset) {
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    /**
     * Returns the total number of elements in the tensor.
     *
     * @return element count
     */
    public abstract int size();

    /**
     * Returns the float value at the given index.
     *
     * @param index element index
     * @return value
     */
    public abstract float getFloat(int index);

    /**
     * Sets the float value at the given index.
     *
     * @param index element index
     * @param value value to set
     */
    public abstract void setFloat(int index, float value);

    /**
     * Returns a float vector view of the data at the given offset.
     *
     * @param species vector species
     * @param offset element offset
     * @return vector of floats
     */
    public abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

    /**
     * Returns the GGML type for this tensor.
     *
     * @return GGML type
     */
    public abstract GGMLType type();

    /**
     * Calculates number of elements from dimensions.
     *
     * @param dimensions tensor dimensions
     * @return total element count
     */
    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    /**
     * Computes a dot product with scalar operations.
     *
     * @param thiz left tensor
     * @param thisOffset left offset
     * @param that right tensor
     * @param thatOffset right offset
     * @param size number of elements
     * @return dot product
     */
    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    /**
     * Computes dot product between two tensors.
     *
     * @param thisOffset offset in this tensor
     * @param that other tensor
     * @param thatOffset offset in other tensor
     * @param size number of elements
     * @return dot product
     */
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    /**
     * Matrix multiplication for a single input vector.
     *
     * @param that vector tensor
     * @param out output tensor
     * @param dim0 output dimension
     * @param dim1 input dimension
     */
    public void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }

    /**
     * Matrix multiplication for multiple input vectors (context).
     *
     * @param context number of vectors
     * @param that input tensor array
     * @param out output tensor array
     * @param dim0 output dimension
     * @param dim1 input dimension
     */
    public void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, (long) dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1));
        });
    }

    /**
     * Aggregation function for reductions.
     */
    @FunctionalInterface
    public interface AggregateFunction {
        /**
         * Applies the aggregation.
         *
         * @param acc accumulated value
         * @param value next value
         * @return new accumulated value
         */
        float apply(float acc, float value);
    }

    /**
     * Reduces a slice of the tensor.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @param seed initial accumulator
     * @param reduce reduce function
     * @return reduced value
     */
    public float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    /**
     * Sums a slice of the tensor.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @return sum of values
     */
    public float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    /**
     * Finds max value in a slice of the tensor.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @return maximum value
     */
    public float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    /**
     * Copies values into another tensor.
     *
     * @param thisOffset offset in this tensor
     * @param that destination tensor
     * @param thatOffset offset in destination tensor
     * @param size number of elements
     */
    public void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    /**
     * Returns the index of the maximum value.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @return index of maximum value
     */
    public int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Returns the index of the maximum value in the entire tensor.
     *
     * @return index of maximum value
     */
    public int argmax() {
        return argmax(0, size());
    }

    /**
     * Mapping function that transforms values.
     */
    @FunctionalInterface
    public interface MapFunction {
        /**
         * Applies transformation to a value.
         *
         * @param value current value
         * @return new value
         */
        float apply(float value);
    }

    /**
     * Mapping function that transforms values with index info.
     */
    @FunctionalInterface
    public interface MapWithIndexFunction {
        /**
         * Applies transformation to a value with its index.
         *
         * @param value current value
         * @param index value index
         * @return new value
         */
        float apply(float value, int index);
    }

    /**
     * Applies a mapping function in place.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @param map mapping function
     * @return this tensor
     */
    public FloatTensor mapInPlace(int thisOffset, int size, MapFunction map) {
        for (int i = thisOffset; i < thisOffset + size; i++) {
            setFloat(i, map.apply(getFloat(i)));
        }
        return this;
    }

    /**
     * Applies a mapping function with index in place.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @param map mapping function
     * @return this tensor
     */
    public FloatTensor mapWithIndexInPlace(int thisOffset, int size, MapWithIndexFunction map) {
        for (int i = thisOffset; i < thisOffset + size; i++) {
            setFloat(i, map.apply(getFloat(i), i));
        }
        return this;
    }

    /**
     * Applies a mapping function to the entire tensor.
     *
     * @param map mapping function
     * @return this tensor
     */
    public FloatTensor mapInPlace(MapFunction map) {
        return mapInPlace(0, size(), map);
    }

    /**
     * Adds another tensor into this tensor in place.
     *
     * @param that other tensor
     * @return this tensor
     */
    public FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, size());
    }

    /**
     * Adds another tensor into this tensor in place using offsets.
     *
     * @param thisOffset offset in this tensor
     * @param that other tensor
     * @param thatOffset offset in other tensor
     * @param size number of elements
     * @return this tensor
     */
    public FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    /**
     * Adds a scaled tensor to this tensor.
     *
     * @param thisOffset offset in this tensor
     * @param that source tensor
     * @param thatOffset offset in source tensor
     * @param size number of elements
     * @param alpha scaling factor
     * @return this tensor
     */
    public FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float alpha) {
        return mapWithIndexInPlace(thisOffset, size,
                (value, index) -> value + alpha * that.getFloat(index - thisOffset + thatOffset));
    }

    /**
     * Multiplies this tensor elementwise by another tensor.
     *
     * @param that other tensor
     * @return this tensor
     */
    public FloatTensor multiplyInPlace(FloatTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }

    /**
     * Multiplies a slice of this tensor by another tensor using offsets.
     *
     * @param thisOffset offset in this tensor
     * @param that other tensor
     * @param thatOffset offset in other tensor
     * @param size number of elements
     * @return this tensor
     */
    public FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    /**
     * Divides each element by a scalar.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @param scalar scalar divisor
     * @return this tensor
     */
    public FloatTensor divideInPlace(int thisOffset, int size, float scalar) {
        return mapInPlace(thisOffset, size, value -> value / scalar);
    }

    /**
     * Fills this tensor slice with a constant value.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @param value fill value
     * @return this tensor
     */
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, ignored -> value);
    }

    /**
     * Computes a softmax in place.
     *
     * @param thisOffset offset in this tensor
     * @param size number of elements
     * @return this tensor
     */
    public FloatTensor softmaxInPlace(int thisOffset, int size) {
        float max = max(thisOffset, size);
        float sum = mapInPlace(thisOffset, size, value -> (float) Math.exp(value - max))
                .sum(thisOffset, size);
        return mapInPlace(thisOffset, size, value -> value / sum);
    }

    /**
     * Returns a float vector for this tensor.
     *
     * @param values values to pack
     * @param species vector species
     * @param index offset to read from values
     * @return vector of floats
     */
    static FloatVector load(FloatTensor values, VectorSpecies<Float> species, int index) {
        return values.getFloatVector(species, index);
    }

    /**
     * Returns a float vector from a float array.
     *
     * @param values value array
     * @param species vector species
     * @param index offset
     * @return vector of floats
     */
    static FloatVector load(float[] values, VectorSpecies<Float> species, int index) {
        return FloatVector.fromArray(species, values, index);
    }
}
