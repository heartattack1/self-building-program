package com.example.kernel.model;

import java.util.Map;

public record InvariantSpec(
        String id,
        String description,
        InvariantType type,
        Map<String, String> params
) {
}
