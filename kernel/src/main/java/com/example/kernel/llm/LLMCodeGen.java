package com.example.kernel.llm;

import com.example.kernel.CodeGen;
import com.example.kernel.GeneratedSourceBundle;
import com.example.kernel.Plan;
import com.example.kernel.config.LlmConfig;
import com.example.kernel.model.StructuredRequirements;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import java.util.logging.Logger;

public class LLMCodeGen implements CodeGen {
    private static final Logger LOGGER = Logger.getLogger(LLMCodeGen.class.getName());
    private final LLMAdapter adapter;
    private final LlmConfig config;
    private final PromptAssembler promptAssembler;
    private final JsonOutputExtractor extractor;
    private final JsonSchemaValidator validator;

    public LLMCodeGen(LLMAdapter adapter, LlmConfig config) {
        this(adapter, config, new PromptAssembler(), new ObjectMapper(), new JsonSchemaValidator());
    }

    public LLMCodeGen(LLMAdapter adapter, LlmConfig config, PromptAssembler promptAssembler, ObjectMapper mapper,
                      JsonSchemaValidator validator) {
        this.adapter = adapter;
        this.config = config;
        this.promptAssembler = promptAssembler;
        this.extractor = new JsonOutputExtractor(mapper);
        this.validator = validator;
    }

    @Override
    public GeneratedSourceBundle generate(StructuredRequirements requirements, Plan plan, int iteration) {
        String prompt = promptAssembler.assemble(requirements, PromptType.CODEGEN, iteration, JsonSchemaSnippets.codeGenSchema());
        long seed = config.seedFor(requirements.specHash()) + iteration;
        LLMRequest request = new LLMRequest(
                "codegen",
                requirements.specHash(),
                prompt,
                config.maxTokens(),
                config.temperature(),
                seed,
                Map.of(
                        "iteration", String.valueOf(iteration),
                        "planVersionId", plan.versionId(),
                        "implMainClass", plan.implClassName()
                )
        );
        LLMResponse response = adapter.generate(request);
        JsonExtractionResult extraction = extractor.extract(response.text());
        if (!extraction.success()) {
            throw new IllegalStateException("Failed to parse codegen JSON: " + extraction.error());
        }
        JsonSchemaValidator.CodeGenResult result = validator.parseCodeGen(extraction.node());
        if (!result.versionId().equals(plan.versionId())) {
            LOGGER.warning(() -> "CodeGen version mismatch: plan=" + plan.versionId() + " codegen=" + result.versionId());
        }
        return result.sources();
    }
}
