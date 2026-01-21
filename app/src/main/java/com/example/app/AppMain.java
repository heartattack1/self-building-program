package com.example.app;

import com.example.api.AppEntry;
import com.example.kernel.Kernel;
import com.example.kernel.config.KernelConfig;
import java.nio.file.Path;
import java.util.logging.Logger;

public class AppMain implements AppEntry {
    private static final Logger LOGGER = Logger.getLogger(AppMain.class.getName());

    public static void main(String[] args) {
        String specPath = args.length > 0 ? args[0] : "./spec.json";
        String configPath = args.length > 1 ? args[1] : "./config.json";
        int code = new AppMain().run(specPath, configPath);
        System.exit(code);
    }

    @Override
    public int run(String specPath) {
        return run(specPath, "./config.json");
    }

    public int run(String specPath, String configPath) {
        KernelConfig config = KernelConfig.load(Path.of(configPath));
        Kernel kernel = new Kernel(Path.of("./var/registry.json"), config);
        boolean ok = kernel.runIterations(Path.of(specPath));
        LOGGER.info("Kernel run complete. active=" + ok);
        return ok ? 0 : 1;
    }
}
