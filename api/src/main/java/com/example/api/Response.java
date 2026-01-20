package com.example.api;

import java.util.Collections;
import java.util.Map;

public record Response(String output, Map<String, String> meta) {
    public Response(String output) {
        this(output, Collections.emptyMap());
    }
}
