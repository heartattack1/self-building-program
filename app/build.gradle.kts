plugins {
    application
}

dependencies {
    implementation(project(":api"))
    implementation(project(":kernel"))
}

application {
    mainClass.set("com.example.app.AppMain")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
