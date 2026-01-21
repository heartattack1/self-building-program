JAR_FILE = target/llama3-1.0.0-SNAPSHOT.jar
JAVA ?= java

.PHONY: jar test run clean

jar:
	mvn package

test:
	mvn test

run: jar
	$(JAVA) --add-modules jdk.incubator.vector -jar $(JAR_FILE)

clean:
	mvn clean
