package com.example.kernel;

import com.example.kernel.model.StructuredRequirements;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;

public class StubPlanner implements Planner {
    @Override
    public Plan plan(StructuredRequirements requirements, int iteration) {
        long seed = seeded(requirements.specHash());
        Random random = new Random(seed + iteration);
        int version = iteration + 1;
        String implClassName = "com.example.impl.v" + version + ".GeneratedService";
        List<String> tasks = List.of(
                "Analyze functional requirements",
                "Generate implementation class " + implClassName,
                "Ensure invariants are enforced",
                "Prepare self-check hook"
        );
        String versionId = "v" + version + "-" + Integer.toHexString(random.nextInt(1_000_000));
        return new Plan(versionId, implClassName, tasks);
    }

    private long seeded(String specHash) {
        byte[] bytes = specHash.getBytes(StandardCharsets.UTF_8);
        long seed = 0;
        for (byte value : bytes) {
            seed = seed * 31 + value;
        }
        return seed;
    }
}
