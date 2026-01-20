package com.example.kernel.compiler;

import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

public class InMemoryJavaCompiler {
    public CompilationResult compile(Map<String, String> sources) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            return new CompilationResult(false, Map.of(), List.of("No Java compiler available"));
        }
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        StandardJavaFileManager standardFileManager = compiler.getStandardFileManager(diagnostics, null, null);
        MemoryFileManager fileManager = new MemoryFileManager(standardFileManager);

        List<JavaFileObject> compilationUnits = new ArrayList<>();
        for (Map.Entry<String, String> entry : sources.entrySet()) {
            compilationUnits.add(new SourceFileObject(entry.getKey(), entry.getValue()));
        }

        List<String> options = List.of("-classpath", System.getProperty("java.class.path"));
        JavaCompiler.CompilationTask task = compiler.getTask(
                null,
                fileManager,
                diagnostics,
                options,
                null,
                compilationUnits
        );
        boolean success = Boolean.TRUE.equals(task.call());
        Map<String, byte[]> classBytes = fileManager.getAllClassBytes();
        List<String> diagnosticMessages = diagnostics.getDiagnostics().stream()
                .map(d -> d.getKind() + ": " + d.getMessage(null))
                .collect(Collectors.toList());
        return new CompilationResult(success, classBytes, diagnosticMessages);
    }

    private static final class SourceFileObject extends javax.tools.SimpleJavaFileObject {
        private final String source;

        private SourceFileObject(String className, String source) {
            super(URI.create("string:///" + className.replace('.', '/') + JavaFileObject.Kind.SOURCE.extension),
                    JavaFileObject.Kind.SOURCE);
            this.source = source;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return source;
        }
    }

    private static final class ByteArrayJavaFileObject extends javax.tools.SimpleJavaFileObject {
        private final ByteArrayOutputStreamWithBytes outputStream = new ByteArrayOutputStreamWithBytes();

        private ByteArrayJavaFileObject(String className, JavaFileObject.Kind kind) {
            super(URI.create("bytes:///" + className.replace('.', '/') + kind.extension), kind);
        }

        @Override
        public java.io.OutputStream openOutputStream() {
            return outputStream;
        }

        private byte[] getBytes() {
            return outputStream.toByteArray();
        }
    }

    private static final class MemoryFileManager extends javax.tools.ForwardingJavaFileManager<StandardJavaFileManager> {
        private final Map<String, ByteArrayJavaFileObject> compiledClasses = new HashMap<>();

        private MemoryFileManager(StandardJavaFileManager fileManager) {
            super(fileManager);
        }

        @Override
        public JavaFileObject getJavaFileForOutput(Location location, String className, JavaFileObject.Kind kind,
                                                   FileObject sibling) {
            ByteArrayJavaFileObject fileObject = new ByteArrayJavaFileObject(className, kind);
            compiledClasses.put(className, fileObject);
            return fileObject;
        }

        private Map<String, byte[]> getAllClassBytes() {
            Map<String, byte[]> bytes = new HashMap<>();
            for (Map.Entry<String, ByteArrayJavaFileObject> entry : compiledClasses.entrySet()) {
                bytes.put(entry.getKey(), entry.getValue().getBytes());
            }
            return bytes;
        }
    }

    private static final class ByteArrayOutputStreamWithBytes extends java.io.ByteArrayOutputStream {
        public byte[] toByteArray() {
            return super.toByteArray();
        }
    }
}
