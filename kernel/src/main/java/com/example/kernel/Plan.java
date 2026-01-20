package com.example.kernel;

import java.util.List;

public record Plan(String versionId, String implClassName, List<String> tasks) {
}
