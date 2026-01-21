package com.llama4j.model;

import com.llama4j.gguf.GGMLTensorEntry;
import com.llama4j.gguf.GGMLType;
import com.llama4j.gguf.GGUF;
import com.llama4j.tensor.BF16FloatTensor;
import com.llama4j.tensor.F16FloatTensor;
import com.llama4j.tensor.FloatTensor;
import com.llama4j.tensor.Q4_0FloatTensor;
import com.llama4j.tensor.Q4_KFloatTensor;
import com.llama4j.tensor.Q8_0FloatTensor;
import com.llama4j.tensor.RoPE;
import com.llama4j.tokenizer.Tokenizer;
import com.llama4j.tokenizer.TokenizerConfig;
import com.llama4j.tokenizer.TokenizerFactory;
import com.llama4j.tokenizer.Vocabulary;
import com.llama4j.util.Pair;
import com.llama4j.util.Timer;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Factory responsible for loading GGUF models into {@link Llama} instances.
 */
public final class ModelLoader {
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";
    private static final String MODEL_NAME = "llama";

    private static final String LLAMA_3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private ModelLoader() {
    }

    /**
     * Loads a GGUF model from the given path.
     *
     * @param ggufPath model path
     * @param contextLength context length override
     * @param loadWeights whether to load weights
     * @return loaded model
     * @throws IOException when reading fails
     */
    public static Llama loadModel(Path ggufPath, int contextLength, boolean loadWeights) throws IOException {
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        return loadModel(fileChannel, gguf, contextLength, loadWeights);
    }

    /**
     * Loads a GGUF model from a file channel and pre-parsed GGUF metadata.
     *
     * @param fileChannel file channel
     * @param gguf parsed GGUF metadata
     * @param contextLength context length override
     * @param loadWeights whether to load weights
     * @return loaded model
     * @throws IOException when reading fails
     */
    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) throws IOException {
        try (var ignored = Timer.log("Load LlaMa model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            Vocabulary vocabulary = loadVocabulary(metadata);
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            Llama.Configuration config = buildConfiguration(metadata, vocabulary, contextLength);

            Llama.Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            return new Llama(config, tokenizer, weights);
        }
    }

    /**
     * Builds a model from metadata without loading tensor weights.
     *
     * @param metadata GGUF metadata
     * @param contextLength context length override
     * @return model with configuration and tokenizer
     */
    public static Llama buildModelFromMetadata(Map<String, Object> metadata, int contextLength) {
        Vocabulary vocabulary = loadVocabulary(metadata);
        Tokenizer tokenizer = createTokenizer(metadata, vocabulary);
        Llama.Configuration config = buildConfiguration(metadata, vocabulary, contextLength);
        return new Llama(config, tokenizer, null);
    }

    /**
     * Loads Llama weights from GGUF tensor entries.
     *
     * @param tensorEntries tensor entries
     * @param config model configuration
     * @return weights container
     */
    public static Llama.Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        float scaleFactor = 8;
        float loFreqFactor = 1;
        float hiFreqFactor = 3;
        int oldContextLength = 8192;
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        return new Llama.Weights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                toFloatBuffer(tensorEntries.get("output_norm.weight")),
                FloatBuffer.wrap(ropeFreqsReal),
                FloatBuffer.wrap(ropeFreqsImag),
                loadQuantized(tensorEntries.getOrDefault("output.weight", tokenEmbeddings))
        );
    }

    /**
     * Loads the tokenizer vocabulary from metadata.
     *
     * @param metadata GGUF metadata
     * @return vocabulary
     */
    static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_LLAMA_3_MODEL.equals(model)) {
            throw new IllegalArgumentException("expected " + TOKENIZER_LLAMA_3_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    /**
     * Creates the tokenizer from metadata and vocabulary.
     *
     * @param metadata GGUF metadata
     * @param vocabulary vocabulary
     * @return tokenizer instance
     */
    static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts -> new Pair<>(
                        vocabulary.getIndex(parts[0]).orElseThrow(),
                        vocabulary.getIndex(parts[1]).orElseThrow()))
                .toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        Map<String, Integer> specialTokens = IntStream.range(0, specialTokensList.size())
                .boxed()
                .collect(Collectors.toMap(i -> specialTokensList.get(i), i -> baseTokens + i));

        TokenizerConfig tokenizerConfig = TokenizerConfig.builder()
                .vocabulary(vocabulary)
                .merges(merges)
                .regexPattern(LLAMA_3_PATTERN)
                .specialTokens(specialTokens)
                .build();
        return TokenizerFactory.createTokenizer(MODEL_NAME, tokenizerConfig);
    }

    /**
     * Builds the configuration from metadata.
     *
     * @param metadata GGUF metadata
     * @param vocabulary vocabulary
     * @param contextLength context length override
     * @return configuration
     */
    static Llama.Configuration buildConfiguration(Map<String, Object> metadata, Vocabulary vocabulary, int contextLength) {
        Llama.Configuration config = new Llama.Configuration(
                (int) metadata.get("llama.embedding_length"),
                (int) metadata.get("llama.feed_forward_length"),
                (int) metadata.get("llama.block_count"),
                (int) metadata.get("llama.attention.head_count"),
                metadata.containsKey("llama.attention.head_count_kv")
                        ? (int) metadata.get("llama.attention.head_count_kv")
                        : (int) metadata.get("llama.attention.head_count"),
                vocabulary.size(),
                (int) metadata.get("llama.context_length"),
                (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)
        ).withContextLength(contextLength);
        return config;
    }

    /**
     * Loads a quantized tensor as a {@link FloatTensor}.
     *
     * @param entry tensor entry
     * @return float tensor
     */
    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_K -> new Q4_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case BF16 -> new BF16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException(
                    "Quantization format " + ggmlType + " is not supported. Supported: Q4_0, Q4_K, Q8_0, F16, BF16.");
        };
    }

    /**
     * Loads an array of quantized tensors.
     *
     * @param size number of tensors
     * @param getTensorEntry tensor entry supplier
     * @return tensor array
     */
    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    /**
     * Loads an array of float buffers.
     *
     * @param size number of buffers
     * @param getTensorEntry tensor entry supplier
     * @return float buffer array
     */
    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    /**
     * Converts a tensor entry to a float buffer.
     *
     * @param tensorEntry tensor entry
     * @return float buffer
     */
    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}
