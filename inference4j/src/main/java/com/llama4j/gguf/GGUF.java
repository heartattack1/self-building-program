package com.llama4j.gguf;

import com.llama4j.tensor.FloatTensor;
import com.llama4j.util.Pair;
import com.llama4j.util.Timer;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parser and container for GGUF model metadata and tensor index data.
 */
public final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32;
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);

    private int magic;
    private int version;
    private int tensorCount;
    private int alignment;
    private int metadataKeyValueCount;
    private Map<String, Object> metadata;
    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;

    private final ByteBuffer buffer1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer buffer2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer buffer4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer buffer8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);

    /**
     * GGUF metadata value types.
     */
    public enum MetadataValueType {
        UINT8(1),
        INT8(1),
        UINT16(2),
        INT16(2),
        UINT32(4),
        INT32(4),
        FLOAT32(4),
        BOOL(1),
        STRING(-8),
        ARRAY(-8),
        UINT64(8),
        INT64(8),
        FLOAT64(8);

        private final int byteSize;

        MetadataValueType(int byteSize) {
            this.byteSize = byteSize;
        }

        private static final MetadataValueType[] VALUES = values();

        /**
         * Returns the type for the given numeric index.
         *
         * @param index numeric index
         * @return metadata value type
         */
        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }

        /**
         * Returns the byte size for the type.
         *
         * @return byte size
         */
        public int byteSize() {
            return byteSize;
        }
    }

    /**
     * Loads and parses a GGUF model file.
     *
     * @param modelPath path to the GGUF file
     * @return parsed GGUF instance
     * @throws IOException when reading fails
     */
    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ);
             var ignored = Timer.log("Parse " + modelPath)) {
            GGUF gguf = new GGUF();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }

    /**
     * Returns the parsed GGUF metadata key/value pairs.
     *
     * @return metadata map
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Returns tensor info descriptors by tensor name.
     *
     * @return tensor info map
     */
    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    /**
     * Returns the offset of tensor data within the GGUF file.
     *
     * @return tensor data offset
     */
    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    /**
     * Parses the GGUF file contents into metadata and tensor info.
 *
     * @param fileChannel file channel
 * @throws IOException when reading fails
 */
    private void loadModelImpl(FileChannel fileChannel) throws IOException {
        readHeader(fileChannel);
        tensorInfos = new HashMap<>(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUFTensorInfo tensorInfo = readTensorInfo(fileChannel);
            tensorInfos.put(tensorInfo.name, tensorInfo);
        }
        long padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + padding);
        tensorDataOffset = fileChannel.position();
    }

    /**
     * Loads tensor entries from a GGUF file using the tensor metadata.
     *
     * @param fileChannel file channel to read from
     * @param tensorDataOffset start offset for tensor data
     * @param tensorInfos tensor metadata by name
     * @return tensor entry map
     * @throws IOException when reading fails
     */
    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset,
                                                          Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        Arena arena = Arena.ofAuto();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset,
                fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = new HashMap<>(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo info = entry.getValue();
            int numberOfElements = FloatTensor.numberOfElements(info.dimensions());
            int sizeInBytes = Math.toIntExact(info.ggmlType().byteSizeFor(numberOfElements));
            MemorySegment memorySegment = tensorData.asSlice(info.offset(), sizeInBytes);
            tensorEntries.put(info.name(), new GGMLTensorEntry(tensorData, info.name(), info.ggmlType(), info.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    /**
     * Represents a parsed GGUF tensor entry.
     *
     * @param name tensor name
     * @param dimensions tensor dimensions
     * @param ggmlType tensor type
     * @param offset offset within the tensor data section
     */
    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    /**
     * Reads a GGML type from the stream.
 *
     * @param fileChannel file channel
 * @return GGML type
 * @throws IOException when reading fails
 */
    private GGMLType readGGMLType(FileChannel fileChannel) throws IOException {
        int ggmlTypeId = readInt(fileChannel);
        return GGMLType.fromId(ggmlTypeId);
    }

    /**
     * Reads a tensor info record.
 *
     * @param fileChannel file channel
 * @return tensor info
 * @throws IOException when reading fails
 */
    private GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException {
        String name = readString(fileChannel);
        int dimensionsCount = readInt(fileChannel);
        int[] dimensions = new int[dimensionsCount];
        for (int i = 0; i < dimensionsCount; ++i) {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        GGMLType ggmlType = readGGMLType(fileChannel);
        long offset = readLong(fileChannel);
        return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    /**
     * Reads a GGUF string value.
 *
     * @param fileChannel file channel
 * @return decoded string
 * @throws IOException when reading fails
 */
    private String readString(FileChannel fileChannel) throws IOException {
        int len = Math.toIntExact(readLong(fileChannel));
        byte[] bytes = new byte[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Reads a metadata key/value pair.
 *
     * @param fileChannel file channel
 * @return key/value pair
 * @throws IOException when reading fails
 */
    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException {
        String key = readString(fileChannel);
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }

    /**
     * Reads a metadata value entry.
 *
     * @param fileChannel file channel
 * @return parsed value
 * @throws IOException when reading fails
 */
    private Object readMetadataValue(FileChannel fileChannel) throws IOException {
        MetadataValueType valueType = readMetadataValueType(fileChannel);
        return readMetadataValueOfType(valueType, fileChannel);
    }

    /**
     * Reads the GGUF header and metadata.
 *
     * @param fileChannel file channel
 * @throws IOException when reading fails
 */
    private void readHeader(FileChannel fileChannel) throws IOException {
        magic = readInt(fileChannel);
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        version = readInt(fileChannel);
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        tensorCount = Math.toIntExact(readLong(fileChannel));
        metadataKeyValueCount = Math.toIntExact(readLong(fileChannel));
        metadata = new HashMap<>(metadataKeyValueCount);
        for (int i = 0; i < metadataKeyValueCount; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    /**
     * Reads an array metadata value.
 *
     * @param fileChannel file channel
 * @return parsed array value
 * @throws IOException when reading fails
 */
    private Object readArray(FileChannel fileChannel) throws IOException {
        MetadataValueType valueType = readMetadataValueType(fileChannel);
        int len = Math.toIntExact(readLong(fileChannel));
        return switch (valueType) {
            case UINT8, INT8 -> {
                byte[] bytes = new byte[len];
                for (int i = 0; i < len; ++i) {
                    bytes[i] = readByte(fileChannel);
                }
                yield bytes;
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(fileChannel);
                }
                yield shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(fileChannel);
                }
                yield ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(fileChannel);
                }
                yield floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(fileChannel);
                }
                yield booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(fileChannel);
                }
                yield strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(fileChannel);
                }
                yield arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + valueType);
        };
    }

    /**
     * Reads a metadata value with a known type.
 *
     * @param valueType value type
 * @param fileChannel file channel
 * @return parsed value
 * @throws IOException when reading fails
 */
    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(fileChannel);
            case UINT16, INT16 -> readShort(fileChannel);
            case UINT32, INT32 -> readInt(fileChannel);
            case FLOAT32 -> readFloat(fileChannel);
            case UINT64, INT64 -> readLong(fileChannel);
            case FLOAT64 -> readDouble(fileChannel);
            case BOOL -> readBoolean(fileChannel);
            case STRING -> readString(fileChannel);
            case ARRAY -> readArray(fileChannel);
        };
    }

    /**
     * Reads a byte from the file.
 *
     * @param fileChannel file channel
 * @return byte value
 * @throws IOException when reading fails
 */
    private byte readByte(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(buffer1);
        assert bytesRead == 1;
        return buffer1.clear().get(0);
    }

    /**
     * Reads a boolean value.
 *
     * @param fileChannel file channel
 * @return boolean value
 * @throws IOException when reading fails
 */
    private boolean readBoolean(FileChannel fileChannel) throws IOException {
        return readByte(fileChannel) != 0;
    }

    /**
     * Reads a short value.
 *
     * @param fileChannel file channel
 * @return short value
 * @throws IOException when reading fails
 */
    private short readShort(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(buffer2);
        assert bytesRead == 2;
        return buffer2.clear().getShort(0);
    }

    /**
     * Reads an int value.
 *
     * @param fileChannel file channel
 * @return int value
 * @throws IOException when reading fails
 */
    private int readInt(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(buffer4);
        assert bytesRead == 4;
        return buffer4.clear().getInt(0);
    }

    /**
     * Reads a long value.
 *
     * @param fileChannel file channel
 * @return long value
 * @throws IOException when reading fails
 */
    private long readLong(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(buffer8);
        assert bytesRead == 8;
        return buffer8.clear().getLong(0);
    }

    /**
     * Reads a float value.
 *
     * @param fileChannel file channel
 * @return float value
 * @throws IOException when reading fails
 */
    private float readFloat(FileChannel fileChannel) throws IOException {
        return Float.intBitsToFloat(readInt(fileChannel));
    }

    /**
     * Reads a double value.
 *
     * @param fileChannel file channel
 * @return double value
 * @throws IOException when reading fails
 */
    private double readDouble(FileChannel fileChannel) throws IOException {
        return Double.longBitsToDouble(readLong(fileChannel));
    }

    /**
     * Reads the metadata value type.
 *
     * @param fileChannel file channel
 * @return metadata value type
 * @throws IOException when reading fails
 */
    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }

    /**
     * Returns the alignment used by the GGUF file.
     *
     * @return alignment in bytes
     */
    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}
