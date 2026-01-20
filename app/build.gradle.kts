plugins {
    application
}

dependencies {
    implementation(project(":api"))
    implementation(project(":kernel"))
}

application {
    mainClass.set("com.example.app.AppMain")
}
