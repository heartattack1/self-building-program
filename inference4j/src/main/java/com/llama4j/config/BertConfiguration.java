package com.llama4j.config;

/**
 * Configuration for BERT style models.
 */
public final class BertConfiguration extends ModelConfiguration {
    private final int vocabularySize;
    private final int contextLength;
    private final int numberOfLayers;
    private final int numberOfHeads;
    private final int hiddenDim;

    /**
     * Creates a BERT configuration.
     *
     * @param vocabularySize vocabulary size
     * @param contextLength context length
     * @param numberOfLayers number of encoder layers
     * @param numberOfHeads number of attention heads
     * @param hiddenDim hidden dimension size
     */
    public BertConfiguration(int vocabularySize, int contextLength, int numberOfLayers, int numberOfHeads, int hiddenDim) {
        this.vocabularySize = vocabularySize;
        this.contextLength = contextLength;
        this.numberOfLayers = numberOfLayers;
        this.numberOfHeads = numberOfHeads;
        this.hiddenDim = hiddenDim;
    }

    @Override
    public String modelName() {
        return "bert";
    }

    @Override
    public int vocabularySize() {
        return vocabularySize;
    }

    @Override
    public int contextLength() {
        return contextLength;
    }

    public int numberOfLayers() {
        return numberOfLayers;
    }

    public int numberOfHeads() {
        return numberOfHeads;
    }

    public int hiddenDim() {
        return hiddenDim;
    }
}
