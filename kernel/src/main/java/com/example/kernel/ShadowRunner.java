package com.example.kernel;

import com.example.api.ServiceFacade;
import com.example.kernel.model.ExampleSpec;
import com.example.kernel.model.InvariantSpec;
import com.example.kernel.model.InvariantType;
import com.example.kernel.model.StructuredRequirements;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

public class ShadowRunner {
    public ShadowReport run(StructuredRequirements requirements,
                            ServiceFacade current,
                            ServiceFacade candidate,
                            boolean requireBackwardCompat) {
        List<String> mismatches = new ArrayList<>();
        List<String> corpus = buildCorpus(requirements);

        for (String input : corpus) {
            String candidateOut = candidate.process(input);
            String candidateOut2 = candidate.process(input);
            String currentOut = current != null ? current.process(input) : null;

            applyInvariants(requirements.invariants(), input, candidateOut, candidateOut2, mismatches);
            applyExamples(requirements.examples(), input, candidateOut, mismatches);

            if (requireBackwardCompat && currentOut != null && !currentOut.equals(candidateOut)) {
                mismatches.add("Backward compatibility mismatch for input " + input);
            }
        }
        return new ShadowReport(mismatches.isEmpty(), mismatches);
    }

    private List<String> buildCorpus(StructuredRequirements requirements) {
        List<String> inputs = new ArrayList<>();
        for (ExampleSpec example : requirements.examples()) {
            inputs.add(example.input());
        }
        long seed = seeded(requirements.specHash());
        Random random = new Random(seed);
        for (int i = 0; i < 20; i++) {
            inputs.add("syn-" + Math.abs(random.nextInt(10_000)));
        }
        return inputs;
    }

    private void applyInvariants(List<InvariantSpec> invariants,
                                 String input,
                                 String output,
                                 String output2,
                                 List<String> mismatches) {
        for (InvariantSpec invariant : invariants) {
            if (invariant.type() == InvariantType.NON_NULL && output == null) {
                mismatches.add(invariant.id() + " failed: output null for input " + input);
            }
            if (invariant.type() == InvariantType.DETERMINISM && output != null && !output.equals(output2)) {
                mismatches.add(invariant.id() + " failed: non-deterministic output for input " + input);
            }
            if (invariant.type() == InvariantType.CONTAINS_SUBSTRING) {
                String substring = invariant.params().get("substring");
                if (substring != null && (output == null || !output.contains(substring))) {
                    mismatches.add(invariant.id() + " failed: output missing substring " + substring);
                }
            }
            if (invariant.type() == InvariantType.REGEX) {
                String regex = invariant.params().get("regex");
                if (regex != null && (output == null || !Pattern.compile(regex).matcher(output).find())) {
                    mismatches.add(invariant.id() + " failed: output does not match regex " + regex);
                }
            }
        }
    }

    private void applyExamples(List<ExampleSpec> examples, String input, String output, List<String> mismatches) {
        for (ExampleSpec example : examples) {
            if (example.input().equals(input)) {
                for (String expected : example.expectedOutputContains()) {
                    if (output == null || !output.contains(expected)) {
                        mismatches.add(example.id() + " failed: output missing expected substring " + expected);
                    }
                }
            }
        }
    }

    private long seeded(String specHash) {
        byte[] bytes = specHash.getBytes(StandardCharsets.UTF_8);
        long seed = 0;
        for (byte value : bytes) {
            seed = seed * 37 + value;
        }
        return seed;
    }
}
