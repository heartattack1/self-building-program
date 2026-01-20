package com.example.kernel;

import com.example.kernel.model.SpecDocument;
import com.example.kernel.model.StructuredRequirements;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.logging.Logger;

public class SpecParser {
    private static final Logger LOGGER = Logger.getLogger(SpecParser.class.getName());
    private final ObjectMapper mapper;

    public SpecParser() {
        this.mapper = new ObjectMapper();
        this.mapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
    }

    public StructuredRequirements parse(Path specPath) throws IOException {
        byte[] bytes = Files.readAllBytes(specPath);
        String specHash = sha256(bytes);
        SpecDocument document = mapper.readValue(bytes, SpecDocument.class);
        validate(document);
        LOGGER.info(() -> "Parsed spec with hash " + specHash);
        return new StructuredRequirements(
                specHash,
                document.meta(),
                document.functionalRequirements(),
                document.invariants(),
                document.examples(),
                document.constraints()
        );
    }

    private void validate(SpecDocument document) {
        if (document.meta() == null) {
            throw new IllegalArgumentException("spec.meta is required");
        }
        if (document.functionalRequirements() == null || document.functionalRequirements().isEmpty()) {
            throw new IllegalArgumentException("spec.functional_requirements is required");
        }
        if (document.invariants() == null) {
            throw new IllegalArgumentException("spec.invariants is required");
        }
        if (document.examples() == null) {
            throw new IllegalArgumentException("spec.examples is required");
        }
        if (document.constraints() == null) {
            throw new IllegalArgumentException("spec.constraints is required");
        }
    }

    private String sha256(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }
}
