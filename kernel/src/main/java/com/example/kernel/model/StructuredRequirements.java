package com.example.kernel.model;

import java.util.List;

public record StructuredRequirements(
        String specHash,
        Meta meta,
        List<FunctionalRequirement> functionalRequirements,
        List<InvariantSpec> invariants,
        List<ExampleSpec> examples,
        ConstraintSpec constraints
) {
}
