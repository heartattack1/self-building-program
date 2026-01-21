package com.llama4j.tensor;

import com.llama4j.gguf.GGMLType;

import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Q4KFloatTensorTest {
    @Test
    void readsPackedValues() {
        byte[] buffer = new byte[GGMLType.Q4_K.getTypeSize()];
        ByteBuffer data = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);
        data.putShort(0, Float.floatToFloat16(1.0f));
        data.putShort(2, Float.floatToFloat16(0.0f));

        int[] scaleValues = new int[16];
        for (int i = 0; i < 8; i++) {
            scaleValues[i] = 1;
        }
        pack6BitValues(buffer, 4, scaleValues);

        byte[] qdata = new byte[GGMLType.QK_K / 2];
        qdata[0] = (byte) ((3 << 4) | 2);
        qdata[1] = (byte) ((5 << 4) | 4);
        System.arraycopy(qdata, 0, buffer, 4 + 12, qdata.length);

        Q4_KFloatTensor tensor = new Q4_KFloatTensor(GGMLType.QK_K, MemorySegment.ofArray(buffer));

        assertEquals(2.0f, tensor.getFloat(0), 1e-6f);
        assertEquals(3.0f, tensor.getFloat(1), 1e-6f);
        assertEquals(4.0f, tensor.getFloat(2), 1e-6f);
        assertEquals(5.0f, tensor.getFloat(3), 1e-6f);

        ArrayFloatTensor dense = new ArrayFloatTensor(new float[]{1f, 1f, 1f, 1f});
        assertEquals(14.0f, tensor.dot(0, dense, 0, 4), 1e-6f);
    }

    private static void pack6BitValues(byte[] buffer, int offset, int[] values) {
        int bitPosition = 0;
        for (int value : values) {
            int byteIndex = offset + bitPosition / 8;
            int bitIndex = bitPosition % 8;
            int current = (buffer[byteIndex] & 0xFF) | ((value & 0x3F) << bitIndex);
            buffer[byteIndex] = (byte) current;
            if (bitIndex > 2) {
                int next = (buffer[byteIndex + 1] & 0xFF) | ((value & 0x3F) >> (8 - bitIndex));
                buffer[byteIndex + 1] = (byte) next;
            }
            bitPosition += 6;
        }
    }
}
