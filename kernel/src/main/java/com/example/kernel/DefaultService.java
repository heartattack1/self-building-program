package com.example.kernel;

import com.example.api.Request;
import com.example.api.Response;
import com.example.api.ServiceFacade;

public class DefaultService implements ServiceFacade {
    @Override
    public Response process(Request request) {
        String input = request != null ? request.input() : null;
        return new Response("OK:" + String.valueOf(input) + ":default");
    }
}
