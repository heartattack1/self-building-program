package com.example.kernel;

import java.util.Collections;
import java.util.List;

public record TestReport(boolean passed, List<String> details) {
    public TestReport {
        details = Collections.unmodifiableList(details);
    }
}
