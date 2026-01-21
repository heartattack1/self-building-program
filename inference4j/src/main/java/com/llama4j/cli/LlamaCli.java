package com.llama4j.cli;

import com.llama4j.aot.AOT;
import com.llama4j.config.LlamaDefaults;
import com.llama4j.model.Llama;
import com.llama4j.model.ModelLoader;
import com.llama4j.sampling.CategoricalSampler;
import com.llama4j.sampling.Sampler;
import com.llama4j.sampling.ToppSampler;
import com.llama4j.tokenizer.ChatFormat;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * CLI entry point for running Llama inference.
 */
public final class LlamaCli {
    /**
     * Batch size used in prompt evaluation.
     */
    private static final int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);

    private LlamaCli() {
    }

    /**
     * Selects the sampler strategy for generation.
     *
     * @param vocabularySize vocabulary size
     * @param temperature sampling temperature
     * @param topp top-p threshold
     * @param rngSeed random seed
     * @return sampler strategy
     */
    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            sampler = Sampler.ARGMAX;
        } else {
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if (topp <= 0 || topp >= 1) {
                innerSampler = new CategoricalSampler(rng);
            } else {
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                logits.divideInPlace(0, logits.size(), temperature);
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    /**
     * Runs the chat loop in interactive mode.
     *
     * @param model model to run
     * @param sampler sampler strategy
     * @param options CLI options
     */
    static void runInteractive(Llama model, Sampler sampler, Options options) {
        Llama.State state = null;
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        conversationTokens.add(chatFormat.getBeginOfTextToken());
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        loop: while (true) {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            switch (userText) {
                case "/quit", "/exit" -> {
                    break loop;
                }
                case "/context" -> {
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(), options.maxTokens(), options.maxTokens() - conversationTokens.size());
                    continue;
                }
                default -> {
                }
            }
            if (state == null) {
                state = model.createNewState(BATCH_SIZE);
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition,
                    conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens,
                    options.maxTokens(), sampler, options.echo(), token -> {
                        if (options.stream()) {
                            if (!model.tokenizer().isSpecialToken(token)) {
                                System.out.print(model.tokenizer().decode(List.of(token)));
                            }
                        }
                    });
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(responseTokens);
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }

    /**
     * Runs a single instruct prompt.
     *
     * @param model model to run
     * @param sampler sampler strategy
     * @param options CLI options
     */
    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        Llama.State state = model.createNewState(BATCH_SIZE);
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());

        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfTextToken());
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens,
                options.maxTokens(), sampler, options.echo(), token -> {
                    if (options.stream()) {
                        if (!model.tokenizer().isSpecialToken(token)) {
                            System.out.print(model.tokenizer().decode(List.of(token)));
                        }
                    }
                });
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }

    /**
     * Program entry point.
     *
     * @param args CLI arguments
     * @throws IOException when loading fails
     */
    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Llama model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
        }
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }

    /**
     * Parsed CLI options.
     *
     * @param modelPath model path
     * @param prompt prompt text
     * @param systemPrompt system prompt
     * @param interactive whether to run in chat mode
     * @param temperature temperature
     * @param topp top-p
     * @param seed RNG seed
     * @param maxTokens maximum tokens
     * @param stream stream output flag
     * @param echo echo prompt flag
     */
    public record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
                          float temperature, float topp, long seed, int maxTokens,
                          boolean stream, boolean echo) {

        /**
         * Validates CLI options.
         */
        public Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null,
                    "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        /**
         * Parses CLI options from arguments.
         *
         * @param args CLI arguments
         * @return parsed options
         */
        public static Options parseOptions(String[] args) {
            boolean interactive = false;
            String prompt = null;
            String systemPrompt = null;
            float temperature = 1.0f;
            float topp = 0.9f;
            long seed = System.currentTimeMillis();
            int maxTokens = LlamaDefaults.DEFAULT_MAX_TOKENS;
            boolean stream = true;
            boolean echo = false;
            Path modelPath = null;

            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (arg.equals("--help") || arg.equals("-h")) {
                    printUsage(System.out);
                    System.exit(0);
                } else if (arg.startsWith("-")) {
                    if (arg.equals("--interactive") || arg.equals("--chat") || arg.equals("-i")) {
                        interactive = true;
                    } else if (arg.equals("--instruct")) {
                        interactive = false;
                    } else {
                        String optionName = arg;
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1;
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Paths.get(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                            case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed,
                    maxTokens, stream, echo);
        }

        /**
         * Ensures a condition holds for CLI parsing.
         *
         * @param condition condition to check
         * @param messageFormat message format
         * @param args format args
         */
        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        /**
         * Prints CLI usage information.
         *
         * @param out output stream
         */
        static void printUsage(PrintStream out) {
            out.println("Usage:  java -jar llama3.jar [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --system-prompt, -sp <string> (optional) system prompt");
            out.println("  --temperature, --temp <f>     (optional) sampling temperature, default = 1.0");
            out.println("  --top-p <f>                   (optional) top-p sampling threshold, default = 0.9");
            out.println("  --seed, -s <int>              (optional) random seed, default = current time millis");
            out.println("  --max-tokens, -n <int>        (optional) maximum generated tokens, default = 512");
            out.println("  --stream <boolean>            (optional) stream output, default = true");
            out.println("  --echo <boolean>              (optional) echo prompt and generated tokens, default = false");
        }
    }
}
