package com.example.kernel;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import org.junit.platform.engine.discovery.DiscoverySelectors;
import org.junit.platform.launcher.Launcher;
import org.junit.platform.launcher.LauncherDiscoveryRequest;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;
import org.junit.platform.launcher.listeners.SummaryGeneratingListener;

public class TestRunner {
    private static final Logger LOGGER = Logger.getLogger(TestRunner.class.getName());

    public TestReport runKernelTests() {
        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                .selectors(DiscoverySelectors.selectPackage("com.example.kernel"))
                .build();
        Launcher launcher = LauncherFactory.create();
        SummaryGeneratingListener listener = new SummaryGeneratingListener();
        launcher.registerTestExecutionListeners(listener);
        launcher.execute(request);
        boolean passed = listener.getSummary().getTotalFailureCount() == 0;
        List<String> details = new ArrayList<>();
        listener.getSummary().getFailures()
                .forEach(failure -> details.add(failure.getTestIdentifier().getDisplayName() + ": " + failure.getException()));
        return new TestReport(passed, details);
    }

    public TestReport runGeneratedSelfCheck(ClassLoader classLoader, String implMainClass) {
        List<String> details = new ArrayList<>();
        try {
            Class<?> implClass = classLoader.loadClass(implMainClass);
            Method method = implClass.getDeclaredMethod("selfCheck");
            method.invoke(null);
            LOGGER.info("Self-check executed for " + implMainClass);
            return new TestReport(true, details);
        } catch (NoSuchMethodException e) {
            details.add("No selfCheck method present");
            return new TestReport(true, details);
        } catch (Exception e) {
            details.add("Self-check failed: " + e.getMessage());
            return new TestReport(false, details);
        }
    }
}
