package com.example.kernel.model;

import java.util.List;

public record ExampleSpec(String id, String input, List<String> expectedOutputContains) {
}
