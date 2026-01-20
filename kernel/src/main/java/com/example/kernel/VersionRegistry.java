package com.example.kernel;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

public class VersionRegistry {
    private static final Logger LOGGER = Logger.getLogger(VersionRegistry.class.getName());
    private final Path registryPath;
    private final ObjectMapper mapper;

    public VersionRegistry(Path registryPath) {
        this.registryPath = registryPath;
        this.mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void append(RegistryRecord record) {
        try {
            Files.createDirectories(registryPath.getParent());
            List<RegistryRecord> records = loadAll();
            records.add(record);
            mapper.writeValue(registryPath.toFile(), records);
        } catch (IOException e) {
            LOGGER.warning("Failed to append registry record: " + e.getMessage());
        }
    }

    public Optional<RegistryRecord> loadLastGood() {
        List<RegistryRecord> records = loadAll();
        Collections.reverse(records);
        return records.stream()
                .filter(record -> "accepted".equals(record.decision()))
                .findFirst();
    }

    private List<RegistryRecord> loadAll() {
        if (!Files.exists(registryPath)) {
            return new ArrayList<>();
        }
        try {
            return mapper.readValue(registryPath.toFile(), new TypeReference<>() {
            });
        } catch (IOException e) {
            LOGGER.warning("Failed to read registry: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    public RegistryRecord buildRecord(String versionId,
                                      String specHash,
                                      Plan plan,
                                      VerificationReport sourceVerification,
                                      VerificationReport bytecodeVerification,
                                      CompilationResultWrapper compilation,
                                      TestReport testReport,
                                      ShadowReport shadowReport,
                                      String decision,
                                      String errorMessage) {
        return new RegistryRecord(
                versionId,
                Instant.now(),
                specHash,
                plan.tasks(),
                sourceVerification,
                bytecodeVerification,
                compilation.diagnostics(),
                testReport,
                shadowReport,
                decision,
                truncate(errorMessage)
        );
    }

    private String truncate(String message) {
        if (message == null) {
            return null;
        }
        if (message.length() <= 500) {
            return message;
        }
        return message.substring(0, 500) + "...";
    }

    public record CompilationResultWrapper(List<String> diagnostics) {
    }
}
