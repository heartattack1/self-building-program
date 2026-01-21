package com.example.kernel.llm;

import com.example.kernel.model.ConstraintSpec;
import com.example.kernel.model.ExampleSpec;
import com.example.kernel.model.FunctionalRequirement;
import com.example.kernel.model.StructuredRequirements;
import java.util.List;
import java.util.stream.Collectors;

public class PromptAssembler {
    public String assemble(StructuredRequirements requirements, PromptType type, int iteration, String schemaSnippet) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("You are an offline code generation model. ");
        prompt.append("Output MUST be strict JSON only, no markdown, no prose, no comments.\n");
        prompt.append("Task: ").append(type.name()).append("\n");
        prompt.append("Spec hash: ").append(requirements.specHash()).append("\n");
        prompt.append("Iteration: ").append(iteration).append("\n\n");

        prompt.append("Meta:\n");
        prompt.append("- name: ").append(requirements.meta().name()).append("\n");
        prompt.append("- version: ").append(requirements.meta().version()).append("\n");
        prompt.append("- description: ").append(requirements.meta().description()).append("\n\n");

        prompt.append("Functional requirements:\n");
        for (FunctionalRequirement requirement : requirements.functionalRequirements()) {
            prompt.append("- ").append(requirement.id()).append(": ").append(requirement.title()).append("\n");
            prompt.append("  ").append(requirement.description()).append("\n");
            if (requirement.acceptanceCriteria() != null && !requirement.acceptanceCriteria().isEmpty()) {
                prompt.append("  acceptance: ")
                        .append(String.join("; ", requirement.acceptanceCriteria()))
                        .append("\n");
            }
        }

        prompt.append("\nInvariants:\n");
        requirements.invariants().forEach(invariant -> prompt.append("- ").append(invariant.description()).append("\n"));

        prompt.append("\nExamples:\n");
        for (ExampleSpec example : requirements.examples()) {
            String expected = example.expectedOutputContains() == null
                    ? "[]"
                    : example.expectedOutputContains().stream().collect(Collectors.joining(", ", "[", "]"));
            prompt.append("- input: ")
                    .append(example.input())
                    .append(" -> output contains: ")
                    .append(expected)
                    .append("\n");
        }

        prompt.append("\nConstraints:\n");
        prompt.append(renderConstraints(requirements.constraints())).append("\n\n");

        prompt.append("Schema:\n").append(schemaSnippet).append("\n");
        prompt.append("Remember: JSON only.");
        return prompt.toString();
    }

    private String renderConstraints(ConstraintSpec constraints) {
        if (constraints == null) {
            return "No constraints provided.";
        }
        return "allowed_packages=" + joinList(constraints.allowedPackages())
                + ", forbidden_packages=" + joinList(constraints.forbiddenPackages())
                + ", forbidden_classes=" + joinList(constraints.forbiddenClasses());
    }

    private String joinList(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "[]";
        }
        return values.stream().collect(Collectors.joining(", ", "[", "]"));
    }
}
