package com.example.kernel;

import com.example.api.Request;
import com.example.api.Response;
import com.example.api.ServiceFacade;
import com.example.kernel.model.StructuredRequirements;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class StubCodeGen implements CodeGen {
    @Override
    public GeneratedSourceBundle generate(StructuredRequirements requirements, Plan plan, int iteration) {
        String packageName = plan.implClassName().substring(0, plan.implClassName().lastIndexOf('.'));
        String className = plan.implClassName().substring(plan.implClassName().lastIndexOf('.') + 1);
        String helperClassName = "Helper";
        String poisonInput = derivePoison(requirements.specHash());
        boolean introduceBug = iteration == 0;

        Map<String, String> sources = new HashMap<>();
        String helperFqcn = packageName + "." + helperClassName;
        sources.put(helperFqcn, helperSource(packageName, helperClassName));
        sources.put(plan.implClassName(), serviceSource(packageName, className, helperClassName, poisonInput, introduceBug));
        return new GeneratedSourceBundle(sources);
    }

    private String derivePoison(String specHash) {
        byte[] bytes = specHash.getBytes(StandardCharsets.UTF_8);
        int sum = 0;
        for (byte value : bytes) {
            sum += value;
        }
        return "seed-" + Math.abs(sum % 97);
    }

    private String helperSource(String packageName, String helperClassName) {
        return "package " + packageName + ";\n\n"
                + "public class " + helperClassName + " {\n"
                + "    public String suffix() {\n"
                + "        return \"v1\";\n"
                + "    }\n"
                + "}\n";
    }

    private String serviceSource(
            String packageName,
            String className,
            String helperClassName,
            String poisonInput,
            boolean introduceBug
    ) {
        String bugClause = introduceBug
                ? "        if (input != null && input.contains(\"" + poisonInput + "\")) {\n"
                + "            return new Response(\"BAD\");\n"
                + "        }\n"
                : "";
        return "package " + packageName + ";\n\n"
                + "import com.example.api.Request;\n"
                + "import com.example.api.Response;\n"
                + "import com.example.api.ServiceFacade;\n\n"
                + "public class " + className + " implements ServiceFacade {\n"
                + "    private final " + helperClassName + " helper = new " + helperClassName + "();\n\n"
                + "    @Override\n"
                + "    public Response process(Request request) {\n"
                + "        String input = request != null ? request.input() : null;\n"
                + "        " + bugClause
                + "        String output = \"OK:\" + String.valueOf(input) + \":\" + helper.suffix();\n"
                + "        return new Response(output);\n"
                + "    }\n\n"
                + "    public static void selfCheck() {\n"
                + "        // Self-check placeholder for generated implementations.\n"
                + "    }\n"
                + "}\n";
    }
}
