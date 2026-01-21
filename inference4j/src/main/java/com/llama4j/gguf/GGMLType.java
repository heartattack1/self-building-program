package com.llama4j.gguf;

/**
 * GGML tensor data types supported by the loader.
 */
public enum GGMLType {
    F32(Float.BYTES),
    F16(GGMLType.FLOAT16_BYTES),
    Q4_0(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE),
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE),
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),
    IQ2_XXS(Integer.MAX_VALUE),
    IQ2_XS(Integer.MAX_VALUE),
    IQ3_XXS(Integer.MAX_VALUE),
    IQ1_S(Integer.MAX_VALUE),
    IQ4_NL(Integer.MAX_VALUE),
    IQ3_S(Integer.MAX_VALUE),
    IQ2_S(Integer.MAX_VALUE),
    IQ4_XS(Integer.MAX_VALUE),
    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(Integer.MAX_VALUE),
    BF16(GGMLType.BFLOAT16_BYTES),
    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    TQ1_0(Integer.MAX_VALUE),
    TQ2_0(Integer.MAX_VALUE);

    /**
     * Size of a bfloat16 in bytes.
     */
    public static final int BFLOAT16_BYTES = 2;

    /**
     * Size of a float16 in bytes.
     */
    public static final int FLOAT16_BYTES = 2;

    /**
     * Block size constant for k-quantization.
     */
    public static final int QK_K = 256;

    private static final GGMLType[] VALUES = values();

    private final int typeSize;
    private final int blockSize;

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    /**
     * Returns the byte size per block.
     *
     * @return type size in bytes
     */
    public int getTypeSize() {
        return typeSize;
    }

    /**
     * Returns the block size.
     *
     * @return block size
     */
    public int getBlockSize() {
        return blockSize;
    }

    /**
     * Returns the GGML type from an ID.
     *
     * @param id type id
     * @return GGML type
     */
    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    /**
     * Computes the byte size for a given element count.
     *
     * @param numberOfElements element count
     * @return byte size
     */
    public long byteSizeFor(int numberOfElements) {
        long total = numberOfElements * (long) getTypeSize();
        assert total % getBlockSize() == 0;
        return Math.toIntExact(total / getBlockSize());
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
