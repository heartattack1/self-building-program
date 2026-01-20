package com.example.kernel;

import com.example.api.ServiceFacade;
import com.example.kernel.compiler.CompilationResult;
import com.example.kernel.compiler.InMemoryJavaCompiler;
import com.example.kernel.model.StructuredRequirements;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Kernel {
    private static final Logger LOGGER = Logger.getLogger(Kernel.class.getName());
    private final SpecParser specParser;
    private final Planner planner;
    private final CodeGen codeGen;
    private final Verifier verifier;
    private final InMemoryJavaCompiler compiler;
    private final TestRunner testRunner;
    private final ShadowRunner shadowRunner;
    private final VersionRegistry registry;
    private final AtomicReference<ServiceFacade> delegate;
    private final HotSwapManager hotSwapManager;

    public Kernel(Path registryPath) {
        this.specParser = new SpecParser();
        this.planner = new StubPlanner();
        this.codeGen = new StubCodeGen();
        this.verifier = new Verifier();
        this.compiler = new InMemoryJavaCompiler();
        this.testRunner = new TestRunner();
        this.shadowRunner = new ShadowRunner();
        this.registry = new VersionRegistry(registryPath);
        this.delegate = new AtomicReference<>(new DefaultService());
        this.hotSwapManager = new HotSwapManager(delegate);
    }

    public StableFacade stableFacade() {
        return new StableFacade(delegate);
    }

    public boolean runIterations(Path specPath) {
        StructuredRequirements requirements = specParserSafe(specPath);
        if (requirements == null) {
            return false;
        }
        Optional<RegistryRecord> lastGood = registry.loadLastGood();
        lastGood.ifPresent(record -> LOGGER.info(() -> "Last good version: " + record.versionId()));

        boolean accepted = false;
        CandidateHandle previousCandidate = hotSwapManager.active();

        for (int iteration = 0; iteration < 3; iteration++) {
            Plan plan = planner.plan(requirements, iteration);
            GeneratedSourceBundle sources = codeGen.generate(requirements, plan, iteration);

            VerificationReport sourceReport = verifier.verifySources(sources.sources(), requirements.constraints());
            if (!sourceReport.passed()) {
                registry.append(registry.buildRecord(plan.versionId(), requirements.specHash(), plan, sourceReport,
                        new VerificationReport(false, java.util.List.of()),
                        new VersionRegistry.CompilationResultWrapper(java.util.List.of()),
                        new TestReport(false, java.util.List.of()),
                        new ShadowReport(false, java.util.List.of()),
                        "rejected", "Source verification failed"));
                continue;
            }

            CompilationResult compilation = compiler.compile(sources.sources());
            if (!compilation.success()) {
                registry.append(registry.buildRecord(plan.versionId(), requirements.specHash(), plan, sourceReport,
                        new VerificationReport(false, java.util.List.of()),
                        new VersionRegistry.CompilationResultWrapper(compilation.diagnostics()),
                        new TestReport(false, java.util.List.of()),
                        new ShadowReport(false, java.util.List.of()),
                        "rejected", "Compilation failed"));
                continue;
            }

            VerificationReport bytecodeReport = verifier.verifyBytecode(compilation.classBytes(), requirements.constraints());
            if (!bytecodeReport.passed()) {
                registry.append(registry.buildRecord(plan.versionId(), requirements.specHash(), plan, sourceReport,
                        bytecodeReport,
                        new VersionRegistry.CompilationResultWrapper(compilation.diagnostics()),
                        new TestReport(false, java.util.List.of()),
                        new ShadowReport(false, java.util.List.of()),
                        "rejected", "Bytecode verification failed"));
                continue;
            }

            CandidateHandle candidate;
            try {
                candidate = hotSwapManager.loadCandidate(compilation.classBytes(), plan.implClassName(), plan.versionId());
            } catch (Exception e) {
                registry.append(registry.buildRecord(plan.versionId(), requirements.specHash(), plan, sourceReport,
                        bytecodeReport,
                        new VersionRegistry.CompilationResultWrapper(compilation.diagnostics()),
                        new TestReport(false, java.util.List.of()),
                        new ShadowReport(false, java.util.List.of()),
                        "rejected", "Load failed: " + e.getMessage()));
                continue;
            }

            TestReport tests = testRunner.runKernelTests();
            TestReport selfCheck = testRunner.runGeneratedSelfCheck(candidate.classLoader(), plan.implClassName());
            boolean testsPassed = tests.passed() && selfCheck.passed();
            TestReport combined = new TestReport(testsPassed,
                    mergeDetails(tests.details(), selfCheck.details()));

            ShadowReport shadow = shadowRunner.run(requirements, delegate.get(), candidate.instance(), false);

            if (testsPassed && shadow.passed()) {
                hotSwapManager.switchTo(candidate);
                registry.append(registry.buildRecord(plan.versionId(), requirements.specHash(), plan, sourceReport,
                        bytecodeReport,
                        new VersionRegistry.CompilationResultWrapper(compilation.diagnostics()),
                        combined,
                        shadow,
                        "accepted", null));
                accepted = true;
                break;
            } else {
                registry.append(registry.buildRecord(plan.versionId(), requirements.specHash(), plan, sourceReport,
                        bytecodeReport,
                        new VersionRegistry.CompilationResultWrapper(compilation.diagnostics()),
                        combined,
                        shadow,
                        "rejected", "Tests or shadow run failed"));
                hotSwapManager.rollbackTo(previousCandidate);
            }
        }

        if (!accepted) {
            LOGGER.warning("No candidate accepted. Retaining last good implementation.");
        }
        return accepted || delegate.get() != null;
    }

    private StructuredRequirements specParserSafe(Path specPath) {
        try {
            return specParser.parse(specPath);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Failed to parse spec: " + e.getMessage(), e);
            return null;
        }
    }

    private java.util.List<String> mergeDetails(java.util.List<String> first, java.util.List<String> second) {
        java.util.List<String> merged = new java.util.ArrayList<>(first);
        merged.addAll(second);
        return merged;
    }
}
