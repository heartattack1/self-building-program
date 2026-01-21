package com.example.kernel.llm;

import com.example.kernel.config.Inference4jConfig;
import com.example.kernel.config.LlmConfig;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Logger;

public class Inference4jLLMAdapter implements LLMAdapter {
    private static final Logger LOGGER = Logger.getLogger(Inference4jLLMAdapter.class.getName());
    private final InferenceBackend backend;
    private final int timeoutMs;
    private final int maxOutputChars;
    private final double topP;
    private final ExecutorService executor;

    public Inference4jLLMAdapter(LlmConfig config) {
        this(buildBackend(config.inference4j()), config.timeoutMs(), config.maxOutputChars(), config.topP());
    }

    public Inference4jLLMAdapter(InferenceBackend backend, int timeoutMs, int maxOutputChars, double topP) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.timeoutMs = timeoutMs;
        this.maxOutputChars = maxOutputChars;
        this.topP = topP;
        this.executor = Executors.newSingleThreadExecutor(r -> {
            Thread thread = new Thread(r, "inference4j-executor");
            thread.setDaemon(true);
            return thread;
        });
    }

    @Override
    public LLMResponse generate(LLMRequest request) {
        Instant start = Instant.now();
        Future<InferenceResult> future = executor.submit(() -> backend.generate(
                new GenerationRequest(request.prompt(), request.maxTokens(), request.temperature(), topP, request.seed())
        ));
        boolean timeout = false;
        InferenceResult result;
        try {
            result = future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            timeout = true;
            future.cancel(true);
            throw new IllegalStateException("Inference timed out after " + timeoutMs + "ms", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Inference interrupted", e);
        } catch (ExecutionException e) {
            throw new IllegalStateException("Inference failed", e.getCause());
        } finally {
            LOGGER.fine(() -> "Inference completed in " + Duration.between(start, Instant.now()).toMillis() + "ms");
        }

        String text = result.text();
        boolean truncated = false;
        if (text.length() > maxOutputChars) {
            text = text.substring(0, maxOutputChars);
            truncated = true;
        }
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("latencyMs", Duration.between(start, Instant.now()).toMillis());
        metrics.put("generatedTokens", result.generatedTokens());
        metrics.put("timeout", timeout);
        metrics.put("truncated", truncated);
        return new LLMResponse(text, metrics);
    }

    private static InferenceBackend buildBackend(Inference4jConfig config) {
        if (config == null || config.modelPath() == null || config.modelPath().isBlank()) {
            throw new IllegalArgumentException("llm.inference4j.model_path is required for inference4j mode");
        }
        return new Inference4jBackend(Path.of(config.modelPath()), config.contextLength());
    }
}
