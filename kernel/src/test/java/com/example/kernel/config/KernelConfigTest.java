package com.example.kernel.config;

import com.example.kernel.Kernel;
import com.example.kernel.llm.LLMCodeGen;
import com.example.kernel.llm.LLMPlanner;
import com.example.kernel.StubCodeGen;
import com.example.kernel.StubPlanner;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KernelConfigTest {
    @Test
    void loadsDefaultsWhenMissing() {
        KernelConfig config = KernelConfig.load(Path.of("does-not-exist.json"));
        assertEquals(LlmMode.STUB, config.llm().mode());
    }

    @Test
    void selectsAdaptersBasedOnMode() throws Exception {
        Path tempModel = Files.createTempFile("model", ".gguf");
        String json = """
                {
                  "llm": {
                    "mode": "inference4j",
                    "inference4j": {
                      "model_path": "%s"
                    }
                  }
                }
                """.formatted(tempModel.toAbsolutePath());
        Path configPath = Files.createTempFile("config", ".json");
        Files.writeString(configPath, json);

        KernelConfig config = KernelConfig.load(configPath);
        Kernel kernel = new Kernel(Files.createTempFile("registry", ".json"), config);
        assertEquals(LlmMode.INFERENCE4J, config.llm().mode());
        assertTrue(getPlanner(kernel) instanceof LLMPlanner);
        assertTrue(getCodeGen(kernel) instanceof LLMCodeGen);

        Kernel stubKernel = new Kernel(Files.createTempFile("registry", ".json"), KernelConfig.defaults());
        assertTrue(getPlanner(stubKernel) instanceof StubPlanner);
        assertTrue(getCodeGen(stubKernel) instanceof StubCodeGen);
    }

    private Object getPlanner(Kernel kernel) throws Exception {
        Field field = Kernel.class.getDeclaredField("planner");
        field.setAccessible(true);
        return field.get(kernel);
    }

    private Object getCodeGen(Kernel kernel) throws Exception {
        Field field = Kernel.class.getDeclaredField("codeGen");
        field.setAccessible(true);
        return field.get(kernel);
    }
}
