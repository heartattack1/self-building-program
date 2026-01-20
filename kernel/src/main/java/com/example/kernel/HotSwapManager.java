package com.example.kernel;

import com.example.api.ServiceFacade;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

public class HotSwapManager {
    private static final Logger LOGGER = Logger.getLogger(HotSwapManager.class.getName());
    private final AtomicReference<CandidateHandle> active = new AtomicReference<>();
    private final AtomicReference<ServiceFacade> delegate;

    public HotSwapManager(AtomicReference<ServiceFacade> delegate) {
        this.delegate = delegate;
    }

    public CandidateHandle loadCandidate(Map<String, byte[]> classBytes, String implMainClass, String versionId)
            throws ReflectiveOperationException {
        VersionedClassLoader classLoader = new VersionedClassLoader(getClass().getClassLoader(), classBytes);
        Class<?> implClass = classLoader.loadClass(implMainClass);
        Object instance = implClass.getDeclaredConstructor().newInstance();
        if (!(instance instanceof ServiceFacade)) {
            throw new IllegalStateException("Implementation does not implement ServiceFacade");
        }
        return new CandidateHandle(versionId, classLoader, (ServiceFacade) instance);
    }

    public void switchTo(CandidateHandle candidate) {
        active.set(candidate);
        delegate.set(candidate.instance());
        LOGGER.info(() -> "Switched to candidate " + candidate.versionId());
    }

    public void rollbackTo(CandidateHandle previous) {
        if (previous != null) {
            active.set(previous);
            delegate.set(previous.instance());
            LOGGER.warning(() -> "Rolled back to candidate " + previous.versionId());
        }
    }

    public CandidateHandle active() {
        return active.get();
    }
}
