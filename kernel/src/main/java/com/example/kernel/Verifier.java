package com.example.kernel;

import com.example.kernel.model.ConstraintSpec;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Verifier {
    private static final List<String> HARD_FORBIDDEN = List.of(
            "java.lang.Runtime",
            "java.lang.ProcessBuilder",
            "sun.misc.Unsafe",
            "java.lang.reflect"
    );

    public VerificationReport verifySources(Map<String, String> sources, ConstraintSpec constraints) {
        List<String> findings = new ArrayList<>();
        List<String> forbiddenPackages = constraints.forbiddenPackages();
        List<String> forbiddenClasses = constraints.forbiddenClasses();

        for (Map.Entry<String, String> entry : sources.entrySet()) {
            String source = entry.getValue();
            for (String pkg : forbiddenPackages) {
                if (source.contains("import " + pkg) || source.contains(pkg + ".")) {
                    findings.add("Source references forbidden package " + pkg + " in " + entry.getKey());
                }
            }
            for (String cls : forbiddenClasses) {
                if (source.contains(cls)) {
                    findings.add("Source references forbidden class " + cls + " in " + entry.getKey());
                }
            }
            for (String hard : HARD_FORBIDDEN) {
                if (source.contains(hard)) {
                    findings.add("Source references hard forbidden class " + hard + " in " + entry.getKey());
                }
            }
            if (source.contains("Class.forName") || source.contains("getDeclared") || source.contains("invoke(")) {
                findings.add("Source contains reflection usage in " + entry.getKey());
            }
            if (source.contains("Runtime.getRuntime") || source.contains("ProcessBuilder")) {
                findings.add("Source contains process execution in " + entry.getKey());
            }
        }
        return new VerificationReport(findings.isEmpty(), findings);
    }

    public VerificationReport verifyBytecode(Map<String, byte[]> classBytes, ConstraintSpec constraints) {
        List<String> findings = new ArrayList<>();
        Set<String> forbiddenTokens = new HashSet<>();
        constraints.forbiddenPackages().forEach(pkg -> forbiddenTokens.add(pkg.replace('.', '/')));
        forbiddenTokens.addAll(constraints.forbiddenClasses());
        forbiddenTokens.addAll(HARD_FORBIDDEN);
        forbiddenTokens.add("java/io");
        forbiddenTokens.add("java/net");
        forbiddenTokens.add("java/lang/reflect");

        for (Map.Entry<String, byte[]> entry : classBytes.entrySet()) {
            List<String> references = scanConstantPoolUtf8(entry.getValue());
            for (String ref : references) {
                for (String forbidden : forbiddenTokens) {
                    if (ref.contains(forbidden)) {
                        findings.add("Bytecode references forbidden token " + forbidden + " in " + entry.getKey());
                    }
                }
            }
        }
        return new VerificationReport(findings.isEmpty(), findings);
    }

    private List<String> scanConstantPoolUtf8(byte[] classBytes) {
        List<String> constants = new ArrayList<>();
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(classBytes))) {
            in.readInt();
            in.readUnsignedShort();
            in.readUnsignedShort();
            int constantPoolCount = in.readUnsignedShort();
            for (int i = 1; i < constantPoolCount; i++) {
                int tag = in.readUnsignedByte();
                switch (tag) {
                    case 1 -> {
                        int length = in.readUnsignedShort();
                        byte[] bytes = in.readNBytes(length);
                        constants.add(new String(bytes, StandardCharsets.UTF_8));
                    }
                    case 3, 4 -> in.skipBytes(4);
                    case 5, 6 -> {
                        in.skipBytes(8);
                        i++;
                    }
                    case 7, 8, 16 -> in.skipBytes(2);
                    case 9, 10, 11, 12, 18 -> in.skipBytes(4);
                    case 15 -> in.skipBytes(3);
                    case 17 -> in.skipBytes(4);
                    default -> throw new IOException("Unknown constant pool tag " + tag);
                }
            }
        } catch (IOException e) {
            constants.add("<parse-error:" + e.getMessage() + ">");
        }
        return constants;
    }
}
