package com.example.kernel.llm;

import com.llama4j.model.Llama;
import com.llama4j.model.ModelLoader;
import com.llama4j.sampling.CategoricalSampler;
import com.llama4j.sampling.Sampler;
import com.llama4j.sampling.ToppSampler;
import com.llama4j.tokenizer.ChatFormat;
import com.llama4j.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class Inference4jBackend implements InferenceBackend {
    private final Path modelPath;
    private final Integer contextLength;
    private final Object lock = new Object();
    private volatile Llama model;
    private volatile ChatFormat chatFormat;

    public Inference4jBackend(Path modelPath, Integer contextLength) {
        if (modelPath == null) {
            throw new IllegalArgumentException("modelPath is required");
        }
        if (!Files.exists(modelPath)) {
            throw new IllegalArgumentException("Model path does not exist: " + modelPath);
        }
        this.modelPath = modelPath;
        this.contextLength = contextLength;
    }

    @Override
    public InferenceResult generate(GenerationRequest request) {
        Llama llama = ensureLoaded();
        ChatFormat format = chatFormat;
        Tokenizer tokenizer = llama.tokenizer();
        List<ChatFormat.Message> messages = List.of(new ChatFormat.Message(ChatFormat.Role.USER, request.prompt()));
        List<Integer> promptTokens = format.encodeDialogPrompt(true, messages);
        Sampler sampler = selectSampler(llama.configuration().vocabularySize(), request.temperature(), request.topP(), request.seed());

        Llama.State state = llama.createNewState(1);
        Set<Integer> stopTokens = format.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(llama, state, 0, promptTokens, stopTokens,
                request.maxTokens(), sampler, false, null);

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }

        List<Integer> printableTokens = new ArrayList<>();
        for (Integer token : responseTokens) {
            if (!tokenizer.isSpecialToken(token)) {
                printableTokens.add(token);
            }
        }
        String text = tokenizer.decode(printableTokens);
        return new InferenceResult(text, responseTokens.size());
    }

    private Llama ensureLoaded() {
        if (model == null) {
            synchronized (lock) {
                if (model == null) {
                    try {
                        model = ModelLoader.loadModel(modelPath, contextLength != null ? contextLength : -1, true);
                        chatFormat = new ChatFormat(model.tokenizer());
                    } catch (IOException e) {
                        throw new IllegalStateException("Failed to load model: " + modelPath, e);
                    }
                }
            }
        }
        return model;
    }

    private Sampler selectSampler(int vocabularySize, double temperature, double topP, long seed) {
        if (temperature <= 0.0) {
            return Sampler.ARGMAX;
        }
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(seed);
        Sampler innerSampler;
        if (topP <= 0 || topP >= 1) {
            innerSampler = new CategoricalSampler(rng);
        } else {
            innerSampler = new ToppSampler(vocabularySize, (float) topP, rng);
        }
        return logits -> {
            logits.divideInPlace(0, logits.size(), (float) temperature);
            logits.softmaxInPlace(0, logits.size());
            return innerSampler.sampleToken(logits);
        };
    }
}
