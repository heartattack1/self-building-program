package com.example.kernel;

import com.example.api.Request;
import com.example.api.Response;
import com.example.api.ServiceFacade;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class StableFacade implements ServiceFacade {
    private final AtomicReference<ServiceFacade> delegate;

    public StableFacade(AtomicReference<ServiceFacade> delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    @Override
    public Response process(Request request) {
        ServiceFacade facade = delegate.get();
        if (facade == null) {
            return new Response("NO_IMPL");
        }
        return facade.process(request);
    }
}
