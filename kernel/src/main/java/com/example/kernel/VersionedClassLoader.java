package com.example.kernel;

import java.util.Map;

public class VersionedClassLoader extends ClassLoader {
    private final Map<String, byte[]> classBytes;

    public VersionedClassLoader(ClassLoader parent, Map<String, byte[]> classBytes) {
        super(parent);
        this.classBytes = classBytes;
    }

    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        byte[] bytes = classBytes.get(name);
        if (bytes != null) {
            return defineClass(name, bytes, 0, bytes.length);
        }
        return super.findClass(name);
    }
}
