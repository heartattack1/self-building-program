package com.example.kernel;

import com.example.kernel.model.StructuredRequirements;

public interface Planner {
    Plan plan(StructuredRequirements requirements, int iteration);
}
