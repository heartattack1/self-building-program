plugins {
    `java-library`
}

dependencies {
    api(project(":api"))
    implementation(project(":inference4j"))

    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.1")
    implementation("org.junit.platform:junit-platform-launcher:1.10.2")

    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
