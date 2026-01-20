package com.example.kernel;

import java.util.Collections;
import java.util.List;

public record VerificationReport(boolean passed, List<String> findings) {
    public VerificationReport {
        findings = Collections.unmodifiableList(findings);
    }
}
