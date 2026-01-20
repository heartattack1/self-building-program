package com.example.kernel.model;

import java.util.List;

public record ConstraintSpec(
        List<String> forbiddenPackages,
        List<String> forbiddenClasses,
        List<String> allowedPackages
) {
}
