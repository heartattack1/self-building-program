package com.llama4j.tensor;

import com.llama4j.util.Pair;

/**
 * Rotary positional embedding utilities.
 */
public final class RoPE {
    private RoPE() {
    }

    /**
     * Precomputes RoPE frequency coefficients.
     *
     * @param contextLength sequence length
     * @param headSize head dimension size
     * @param theta base theta value
     * @param ropeScaling whether to apply scaling
     * @param scaleFactor scaling factor
     * @param loFreqFactor low frequency factor
     * @param hiFreqFactor high frequency factor
     * @param oldContextLength original context length
     * @return pair of real/imaginary arrays
     */
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
                                                           boolean ropeScaling, float scaleFactor, float loFreqFactor,
                                                           float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
}
