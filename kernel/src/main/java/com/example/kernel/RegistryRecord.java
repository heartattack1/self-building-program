package com.example.kernel;

import java.time.Instant;
import java.util.List;

public record RegistryRecord(
        String versionId,
        Instant timestamp,
        String specHash,
        List<String> planSummary,
        VerificationReport sourceVerification,
        VerificationReport bytecodeVerification,
        List<String> compilationDiagnostics,
        TestReport testReport,
        ShadowReport shadowReport,
        String decision,
        String errorMessage
) {
}
