package com.example.kernel.model;

import java.util.List;

public record FunctionalRequirement(
        String id,
        String title,
        String description,
        List<IoField> inputs,
        List<IoField> outputs,
        List<String> acceptanceCriteria
) {
}
