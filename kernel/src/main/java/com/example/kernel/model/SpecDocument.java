package com.example.kernel.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

public record SpecDocument(
        Meta meta,
        @JsonProperty("functional_requirements") List<FunctionalRequirement> functionalRequirements,
        List<InvariantSpec> invariants,
        List<ExampleSpec> examples,
        ConstraintSpec constraints
) {
}
