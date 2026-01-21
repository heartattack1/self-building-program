package com.llama4j.aot;

import com.llama4j.config.LlamaDefaults;
import com.llama4j.gguf.GGMLTensorEntry;
import com.llama4j.gguf.GGUF;
import com.llama4j.model.Llama;
import com.llama4j.model.ModelLoader;
import com.llama4j.util.Timer;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;

/**
 * Support for AOT preloading of GGUF metadata with GraalVM native-image.
 */
public final class AOT {
    /**
     * Partial model information stored when preloading is enabled.
     *
     * @param modelFileName model file name
     * @param model base model without weights
     * @param tensorDataOffset tensor data offset
     * @param tensorInfos tensor metadata
     */
    record PartialModel(String modelFileName, Llama model, long tensorDataOffset,
                        Map<String, GGUF.GGUFTensorInfo> tensorInfos) {
    }

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("llama.PreloadGGUF"));

    private AOT() {
    }

    /**
     * Attempts to preload a model for native-image builds.
     *
     * @param modelPath path to the GGUF model
     * @return preloaded model info or null
     */
    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            GGUF gguf = GGUF.loadModel(path);
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                return new PartialModel(
                        path.getFileName().toString(),
                        ModelLoader.loadModel(fileChannel, gguf, LlamaDefaults.DEFAULT_MAX_TOKENS, false),
                        gguf.getTensorDataOffset(),
                        gguf.getTensorInfos()
                );
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Tries to reuse a compatible AOT preloaded model.
     *
     * @param modelPath requested model path
     * @param contextLength context length override
     * @return model with loaded weights or null if not compatible
     * @throws IOException when loading fails
     */
    public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        PartialModel preLoaded = PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
            Llama.Weights weights = ModelLoader.loadWeights(tensorEntries, baseModel.configuration());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
        }
    }
}
