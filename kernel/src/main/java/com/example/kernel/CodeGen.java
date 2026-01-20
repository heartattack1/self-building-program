package com.example.kernel;

import com.example.kernel.model.StructuredRequirements;

public interface CodeGen {
    GeneratedSourceBundle generate(StructuredRequirements requirements, Plan plan, int iteration);
}
