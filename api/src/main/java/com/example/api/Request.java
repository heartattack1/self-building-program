package com.example.api;

import java.util.Collections;
import java.util.Map;

public record Request(String input, Map<String, String> attributes) {
    public Request(String input) {
        this(input, Collections.emptyMap());
    }
}
