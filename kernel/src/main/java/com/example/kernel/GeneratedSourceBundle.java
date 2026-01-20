package com.example.kernel;

import java.util.Collections;
import java.util.Map;

public record GeneratedSourceBundle(Map<String, String> sources) {
    public GeneratedSourceBundle {
        sources = Collections.unmodifiableMap(sources);
    }
}
