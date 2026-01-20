package com.example.kernel;

import com.example.api.ServiceFacade;

public record CandidateHandle(String versionId, VersionedClassLoader classLoader, ServiceFacade instance) {
}
