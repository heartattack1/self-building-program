package com.example.kernel.compiler;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public record CompilationResult(boolean success, Map<String, byte[]> classBytes, List<String> diagnostics) {
    public CompilationResult {
        classBytes = Collections.unmodifiableMap(classBytes);
        diagnostics = Collections.unmodifiableList(diagnostics);
    }
}
