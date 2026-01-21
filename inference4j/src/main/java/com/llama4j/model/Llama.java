package com.llama4j.model;

import com.llama4j.config.ModelConfiguration;
import com.llama4j.sampling.Sampler;
import com.llama4j.tensor.ArrayFloatTensor;
import com.llama4j.tensor.FloatTensor;
import com.llama4j.tokenizer.Tokenizer;
import com.llama4j.util.Parallel;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Core Llama model definition, including configuration, weights, and inference routines.
 *
 * @param configuration model configuration
 * @param tokenizer tokenizer instance
 * @param weights model weights
 */
public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
    /**
     * Creates a new mutable state for inference.
     *
     * @param batchsize batch size
     * @return new model state
     */
    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    /**
     * Model configuration parameters.
     */
    public static final class Configuration extends ModelConfiguration {
        /** transformer dimension */
        public final int dim;
        /** FFN hidden dimension */
        public final int hiddenDim;
        /** number of layers */
        public final int numberOfLayers;
        /** number of attention heads */
        public final int numberOfHeads;
        /** number of key/value heads */
        public final int numberOfKeyValueHeads;
        /** vocabulary size */
        public final int vocabularySize;
        /** context length */
        public final int contextLength;
        /** RMSNorm epsilon */
        public final float rmsNormEps;
        /** RoPE base theta */
        public final float ropeTheta;
        /** head size */
        public final int headSize;

        /**
         * Creates a configuration.
         *
         * @param dim transformer dimension
         * @param hiddenDim FFN hidden dimension
         * @param numberOfLayers number of layers
         * @param numberOfHeads number of attention heads
         * @param numberOfKeyValueHeads number of key/value heads
         * @param vocabularySize vocabulary size
         * @param contextLength context length
         * @param rmsNormEps RMS norm epsilon
         * @param ropeTheta RoPE theta
         */
        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                             int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }

        @Override
        public String modelName() {
            return "llama";
        }

        @Override
        public int vocabularySize() {
            return vocabularySize;
        }

        @Override
        public int contextLength() {
            return contextLength;
        }

        /**
         * Returns a copy with updated context length.
         *
         * @param newContextLength new context length
         * @return updated configuration
         */
        public Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this;
            }
            return new Configuration(dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                    vocabularySize, newContextLength, rmsNormEps, ropeTheta);
        }
    }

    /**
     * Container for model weights.
     */
    public static final class Weights {
        /** token embedding table */
        public final FloatTensor token_embedding_table;
        /** RMSNorm weights for attention */
        public final FloatBuffer[] rms_att_weight;
        /** query weights */
        public final FloatTensor[] wq;
        /** key weights */
        public final FloatTensor[] wk;
        /** value weights */
        public final FloatTensor[] wv;
        /** output projection weights */
        public final FloatTensor[] wo;
        /** RMSNorm weights for FFN */
        public final FloatBuffer[] rms_ffn_weight;
        /** FFN gate weights */
        public final FloatTensor[] w1;
        /** FFN down projection weights */
        public final FloatTensor[] w2;
        /** FFN up projection weights */
        public final FloatTensor[] w3;
        /** final RMSNorm weights */
        public final FloatBuffer rms_final_weight;
        /** RoPE real part */
        public final FloatBuffer freq_cis_real;
        /** RoPE imaginary part */
        public final FloatBuffer freq_cis_imag;
        /** classifier weights */
        public final FloatTensor wcls;

        /**
         * Creates the weights container.
         *
         * @param token_embedding_table token embedding table
         * @param rms_att_weight attention RMS weights
         * @param wq query weights
         * @param wk key weights
         * @param wv value weights
         * @param wo output projection weights
         * @param rms_ffn_weight FFN RMS weights
         * @param w1 FFN gate weights
         * @param w2 FFN down projection weights
         * @param w3 FFN up projection weights
         * @param rms_final_weight final RMS weights
         * @param freq_cis_real RoPE real part
         * @param freq_cis_imag RoPE imaginary part
         * @param wcls classifier weights
         */
        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq,
                       FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight,
                       FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight,
                       FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.rms_final_weight = rms_final_weight;
            this.freq_cis_real = freq_cis_real;
            this.freq_cis_imag = freq_cis_imag;
            this.wcls = wcls;
        }
    }

    /**
     * Mutable inference state, including activation buffers and KV caches.
     */
    public static final class State {
        /** batch size */
        public final int batchsize;
        /** activation at current time stamp */
        public final FloatTensor[] x;
        /** activation buffer inside residual branch */
        public final FloatTensor[] xb;
        /** extra buffer for attention output */
        public final FloatTensor[] xb2;
        /** FFN hidden buffer */
        public final FloatTensor[] hb;
        /** second FFN hidden buffer */
        public final FloatTensor[] hb2;
        /** query buffer */
        public final FloatTensor[] q;
        /** key buffer */
        public final FloatTensor[] k;
        /** value buffer */
        public final FloatTensor[] v;
        /** attention scores buffer */
        public final FloatTensor[] att;
        /** output logits */
        public final FloatTensor logits;
        /** key cache */
        public final FloatTensor[] keyCache;
        /** value cache */
        public final FloatTensor[] valueCache;

        /** last index in previous block */
        int idxPrevBlock;

        /** most recently generated token */
        public int latestToken;

        /**
         * Creates a new state for the configuration and batch size.
         *
         * @param config model configuration
         * @param batchsize batch size
         */
        State(Configuration config, int batchsize) {
            this.batchsize = batchsize;
            this.x = allocate(batchsize, config.dim);
            this.xb = allocate(batchsize, config.dim);
            this.xb2 = allocate(batchsize, config.dim);
            this.hb = allocate(batchsize, config.hiddenDim);
            this.hb2 = allocate(batchsize, config.hiddenDim);
            this.q = allocate(batchsize, config.dim);
            this.k = allocate(batchsize, config.dim);
            this.v = allocate(batchsize, config.dim);
            this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);
            idxPrevBlock = -1;

            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim))
                    .limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim))
                    .limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    /**
     * Allocates a batch of dense tensors with the given dimensions.
     *
     * @param numTokens number of tensors
     * @param dims tensor dimensions
     * @return tensor array
     */
    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    /**
     * Applies RMS normalization.
     *
     * @param out output tensor
     * @param x input tensor
     * @param weight RMS weights
     * @param size size of the tensor
     * @param rmsNormEps epsilon value
     */
    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    /**
     * Runs a forward pass for the given tokens.
     *
     * @param model model instance
     * @param state model state
     * @param tokens tokens to process
     * @param position start position in the context
     * @param computeLogits whether to compute logits
     * @return logits or null when skipped
     */
    static FloatTensor forward(Llama model, State state, int[] tokens, int position, boolean computeLogits) {
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads;
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        final int nTokens = tokens.length;

        Parallel.parallelFor(0, nTokens, t ->
                weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim)
        );

        for (int l = 0; l < config.numberOfLayers; l++) {
            final int curLayer = l;
            Parallel.parallelFor(0, nTokens, t ->
                    rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );

            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);

            Parallel.parallelFor(0, nTokens, t -> {
                for (int i = 0; i < dim; i += 2) {
                    int headDim = i % headSize;
                    float fcr = weights.freq_cis_real.get((position + t) * (headSize / 2) + (headDim / 2));
                    float fci = weights.freq_cis_imag.get((position + t) * (headSize / 2) + (headDim / 2));
                    int rotn = i < kvDim ? 2 : 1;
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = vi == 0 ? state.q[t] : state.k[t];
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i, v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
            });

            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                state.idxPrevBlock = nTokens - 1;
                return null;
            }

            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength;

                for (int t = 0; t <= position + token; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q[token].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att[token].setFloat(attOffset + t, score);
                }

                state.att[token].softmaxInPlace(attOffset, position + token + 1);

                int xbOffset = h * headSize;
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position + token; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att[token].getFloat(attOffset + t);
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, dim);

            Parallel.parallelFor(0, nTokens, t -> state.x[t].addInPlace(state.xb2[t]));

            Parallel.parallelFor(0, nTokens, t ->
                    rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps)
            );

            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);

            Parallel.parallelFor(0, nTokens, t ->
                    state.hb[t].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)))
            );

            Parallel.parallelFor(0, nTokens, t -> state.hb[t].multiplyInPlace(state.hb2[t]));

            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);

            Parallel.parallelFor(0, nTokens, t -> state.x[t].addInPlace(state.xb[t]));
        }

        Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps)
        );

        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);
        state.idxPrevBlock = nTokens - 1;

        return state.logits;
    }

    /**
     * Generates tokens for a prompt using the model.
     *
     * @param model model instance
     * @param state model state
     * @param startPosition starting position in context
     * @param promptTokens prompt tokens to ingest
     * @param stopTokens stop token set
     * @param maxTokens maximum token count
     * @param sampler sampling strategy
     * @param echo whether to echo tokens to stderr
     * @param onTokenGenerated callback invoked for each generated token
     * @return list of generated tokens (including stop token if present)
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens,
                                               Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken;
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position,
                        Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    System.out.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n",
                            position, promptIndex, promptTokens.size(), Arrays.toString(tokens));
                }
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
                forward(model, state, tokens, position, computeLogits);
                position += nTokens - 1;
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
                forward(model, state, new int[]{token}, position, true);
            }
            nextToken = sampler.sampleToken(state.logits);
            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        System.err.printf("%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());

        return generatedTokens;
    }
}
