package com.example.api;

public interface ServiceFacade {
    Response process(Request request);

    default String process(String input) {
        Response response = process(new Request(input));
        return response != null ? response.output() : null;
    }
}
