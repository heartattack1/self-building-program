package com.example.kernel.llm;

public final class JsonSchemaSnippets {
    private JsonSchemaSnippets() {
    }

    public static String planSchema() {
        return """
                {
                  "versionId": "string",
                  "implMainClass": "string",
                  "files": [
                    {"fqcn": "string", "role": "string"}
                  ],
                  "invariants": ["string"],
                  "notes": "string"
                }
                """.trim();
    }

    public static String codeGenSchema() {
        return """
                {
                  "versionId": "string",
                  "files": {
                    "com.example.impl.v1.GeneratedService": "java source text"
                  },
                  "notes": "string"
                }
                """.trim();
    }
}
