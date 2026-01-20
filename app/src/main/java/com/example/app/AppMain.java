package com.example.app;

import com.example.api.AppEntry;
import com.example.kernel.Kernel;
import java.nio.file.Path;
import java.util.logging.Logger;

public class AppMain implements AppEntry {
    private static final Logger LOGGER = Logger.getLogger(AppMain.class.getName());

    public static void main(String[] args) {
        String specPath = args.length > 0 ? args[0] : "./spec.json";
        int code = new AppMain().run(specPath);
        System.exit(code);
    }

    @Override
    public int run(String specPath) {
        Kernel kernel = new Kernel(Path.of("./var/registry.json"));
        boolean ok = kernel.runIterations(Path.of(specPath));
        LOGGER.info("Kernel run complete. active=" + ok);
        return ok ? 0 : 1;
    }
}
