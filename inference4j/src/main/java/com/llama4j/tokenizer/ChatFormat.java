package com.llama4j.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
public class ChatFormat {
    private final Tokenizer tokenizer;
    private final int beginOfText;
    private final int endHeader;
    private final int startHeader;
    private final int endOfTurn;
    private final int endOfText;
    private final int endOfMessage;
    private final Set<Integer> stopTokens;

    /**
     * Creates a chat formatter for the provided tokenizer.
     *
     * @param tokenizer tokenizer with special tokens
     */
    public ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1);
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    /**
     * Returns the tokenizer used by this chat formatter.
     *
     * @return tokenizer
     */
    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    /**
     * Returns the stop tokens used for generation.
     *
     * @return stop token set
     */
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    /**
     * Returns the begin-of-text token id.
     *
     * @return begin-of-text token id
     */
    public int getBeginOfTextToken() {
        return beginOfText;
    }

    /**
     * Encodes a message header for a given role.
     *
     * @param message message to encode
     * @return token ids for the header
     */
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }

    /**
     * Encodes a message with header and end-of-turn marker.
     *
     * @param message message to encode
     * @return token ids for the message
     */
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    /**
     * Encodes a dialog prompt, optionally appending the assistant prefix.
     *
     * @param appendAssistantTurn whether to append an assistant header
     * @param dialog list of dialog messages
     * @return token ids for the prompt
     */
    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            tokens.addAll(this.encodeHeader(new Message(Role.ASSISTANT, "")));
        }
        return tokens;
    }

    /**
     * Message container for chat formatting.
     *
     * @param role role name
     * @param content message content
     */
    public record Message(Role role, String content) {
    }

    /**
     * Role names used by the chat format.
     *
     * @param name role name
     */
    public record Role(String name) {
        /** Default system role. */
        public static Role SYSTEM = new Role("system");
        /** Default user role. */
        public static Role USER = new Role("user");
        /** Default assistant role. */
        public static Role ASSISTANT = new Role("assistant");

        /**
         * Returns the role name.
         *
         * @return role name
         */
        @Override
        public String toString() {
            return name;
        }
    }
}
