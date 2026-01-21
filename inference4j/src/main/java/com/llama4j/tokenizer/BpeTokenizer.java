package com.llama4j.tokenizer;

import com.llama4j.util.Pair;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Byte Pair Encoding tokenizer implementation.
 *
 * <p>Based on minbpe and the GPT-2 tokenizer design.</p>
 */
public class BpeTokenizer implements Tokenizer {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    /**
     * Creates a tokenizer from vocabulary, merges, regex pattern and special tokens.
     *
     * @param vocabulary vocabulary entries
     * @param merges list of merge pairs
     * @param regexPattern regex pattern to chunk input text
     * @param specialTokens map of special token strings to indices
     */
    public BpeTokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern,
                        Map<String, Integer> specialTokens) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    /**
     * Returns the regular expression pattern used by the tokenizer.
     *
     * @return regex pattern string or null
     */
    @Override
    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }

    /**
     * Returns the map of special token strings to indices.
     *
     * @return special token map
     */
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    /**
     * Returns true if the token index corresponds to a special token.
     *
     * @param tokenIndex token index
     * @return true if special token
     */
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    /**
     * Encodes a string into token ids using BPE.
     *
     * @param text text to encode
     * @return token ids
     */
    @Override
    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    /**
     * Encodes a string into a list of token ids.
     *
     * @param text text to encode
     * @return list of token ids
     */
    @Override
    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    /**
     * Encodes text while handling special tokens.
     *
     * @param text text to encode
     * @param allowedSpecial set of allowed special token strings
     * @return token ids
     */
    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            return encodeOrdinary(text);
        }

        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                ids.add(getSpecialTokens().get(part));
            } else {
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    /**
     * Encodes text while ignoring special tokens.
     *
     * @param text text to encode
     * @return token ids
     */
    @Override
    public List<Integer> encodeOrdinary(String text) {
        List<String> textChunks = findAll(compiledPattern, text);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    /**
     * Decodes token ids back into a string.
     *
     * @param tokens token ids
     * @return decoded string
     */
    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decoded.length(); i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }

    /**
     * Replaces control characters in code points with escaped sequences.
     *
     * @param codePoints code points to sanitize
     * @return sanitized string
     */
    public Vocabulary getVocabulary() {
        return vocabulary;
    }

    /**
     * Returns the map of merges.
     *
     * @return merge map
     */
    public Map<Pair<Integer, Integer>, Integer> getMerges() {
        return merges;
    }

    /**
     * Internal encode entry point that defaults to ordinary encoding.
     *
     * @param text text to encode
     * @return token ids
     */
    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    /**
     * Finds all regex matches in a string.
     *
     * @param pattern regex pattern to apply
     * @param text input text
     * @return list of matched substrings
     */
    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Computes merge pair statistics for the given token ids.
     *
     * @param ids token ids
     * @return map of pair counts
     */
    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    /**
     * Encodes a chunk of text using byte-level BPE merges.
     *
     * @param chunk input chunk
     * @return list of token ids
     */
    private List<Integer> encodeChunk(String chunk) {
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            if (!this.merges.containsKey(pair)) {
                break;
            }
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    /**
     * Applies a merge operation to token ids.
     *
     * @param ids token ids
     * @param pair pair to merge
     * @param idx merged token id
     * @return merged token ids
     */
    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    /**
     * Decodes token ids into their raw unicode string without byte decoding.
     *
     * @param tokens token ids
     * @return decoded unicode string
     */
    private String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Creates a reversible byte-to-unicode mapping.
     *
     * @return mapping of byte values to unicode code points
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    /**
     * Mapping from byte values to unicode code points.
     */
    public static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();

    /**
     * Mapping from unicode code points to byte values.
     */
    public static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
}
