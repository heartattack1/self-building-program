package com.example.kernel.llm;

import com.example.kernel.Plan;
import com.example.kernel.Planner;
import com.example.kernel.config.LlmConfig;
import com.example.kernel.model.StructuredRequirements;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import java.util.logging.Logger;

public class LLMPlanner implements Planner {
    private static final Logger LOGGER = Logger.getLogger(LLMPlanner.class.getName());
    private final LLMAdapter adapter;
    private final LlmConfig config;
    private final PromptAssembler promptAssembler;
    private final JsonOutputExtractor extractor;
    private final JsonSchemaValidator validator;

    public LLMPlanner(LLMAdapter adapter, LlmConfig config) {
        this(adapter, config, new PromptAssembler(), new ObjectMapper(), new JsonSchemaValidator());
    }

    public LLMPlanner(LLMAdapter adapter, LlmConfig config, PromptAssembler promptAssembler, ObjectMapper mapper,
                      JsonSchemaValidator validator) {
        this.adapter = adapter;
        this.config = config;
        this.promptAssembler = promptAssembler;
        this.extractor = new JsonOutputExtractor(mapper);
        this.validator = validator;
    }

    @Override
    public Plan plan(StructuredRequirements requirements, int iteration) {
        String prompt = promptAssembler.assemble(requirements, PromptType.PLAN, iteration, JsonSchemaSnippets.planSchema());
        long seed = config.seedFor(requirements.specHash()) + iteration;
        LLMRequest request = new LLMRequest(
                "plan",
                requirements.specHash(),
                prompt,
                config.maxTokens(),
                config.temperature(),
                seed,
                Map.of("iteration", String.valueOf(iteration))
        );
        LLMResponse response = adapter.generate(request);
        JsonExtractionResult extraction = extractor.extract(response.text());
        if (!extraction.success()) {
            throw new IllegalStateException("Failed to parse plan JSON: " + extraction.error());
        }
        Plan plan = validator.parsePlan(extraction.node());
        LOGGER.info(() -> "LLM plan generated versionId=" + plan.versionId());
        return plan;
    }
}
