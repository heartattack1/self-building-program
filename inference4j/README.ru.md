# Llama3

## Требования

- Java 25+ (используется `MemorySegment` mmap).
- Maven 3.x

## Скачайте модель (GGUF)

Скачайте файл GGUF в квантизации `Q4_0` или `Q8_0`, например:

```bash
# Llama 3.2 (3B)
curl -L -O https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf

# Llama 3.2 (1B)
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
```

## Сборка

```bash
mvn package
```

## Запуск

```bash
java --add-modules jdk.incubator.vector \
  -jar target/llama3-1.0.0-SNAPSHOT.jar \
  --model /path/to/model.gguf \
  --chat
```

## Запуск из исходников

```bash
mvn -q exec:java \
  -Dexec.mainClass=com.llama4j.cli.LlamaCli \
  -Dexec.args="--model /path/to/model.gguf --chat"
```
