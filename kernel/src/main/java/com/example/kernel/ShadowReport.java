package com.example.kernel;

import java.util.Collections;
import java.util.List;

public record ShadowReport(boolean passed, List<String> mismatches) {
    public ShadowReport {
        mismatches = Collections.unmodifiableList(mismatches);
    }
}
