# Llama3

## Requirements

- Java 25+ (uses `MemorySegment` mmap).
- Maven 3.x

## Architecture overview

This project exposes a flexible, extensible model API:

- `Model` interface: `com.llama4j.model.Model`
- Model factory: `com.llama4j.model.ModelFactory`
- Tokenizer factory: `com.llama4j.tokenizer.TokenizerFactory`
- Configuration base type: `com.llama4j.config.ModelConfiguration`

Each model implementation supplies its own configuration class, tokenizer provider, and a service provider entry in
`src/main/resources/META-INF/services`.

## Adding a new model

1. Create a new `Model` implementation (for example `com.llama4j.model.MyModel`) and a configuration class that extends
   `com.llama4j.config.ModelConfiguration`.
2. Add a `ModelProvider` implementation that returns your model.
3. Implement a `TokenizerProvider` and register it in
   `src/main/resources/META-INF/services/com.llama4j.tokenizer.TokenizerProvider`.
4. Register your `ModelProvider` in
   `src/main/resources/META-INF/services/com.llama4j.model.ModelProvider`.

No changes are needed in the shared factories once the providers are registered.

## Download a model (GGUF)

Download a `Q4_0`, `Q4_K`, or `Q8_0` GGUF file, for example:

```bash
# Llama 3.2 (3B)
curl -L -O https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf

# Llama 3.2 (1B)
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Llama 3.2 (1B) Q4_K_M
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## Build

```bash
mvn package
```

## Tests

```bash
mvn test
```

## Run

```bash
java --add-modules jdk.incubator.vector \
  -jar target/llama3-1.0.0-SNAPSHOT.jar \
  --model /path/to/model.gguf \
  --chat
```

## Examples

## Supported quantization

The loader currently supports `Q4_0`, `Q4_K` (including `Q4_K_M`), `Q8_0`, `F16`, and `BF16` GGUF tensors. Other formats (including `Q5_K`) are not yet supported.

### Single-shot instruct prompt

```bash
java --add-modules jdk.incubator.vector \
  -jar target/llama3-1.0.0-SNAPSHOT.jar \
  --model /path/to/model.gguf \
  --prompt "Explain Java memory segments in one paragraph."
```

### Chat mode with a system prompt

```bash
java --add-modules jdk.incubator.vector \
  -jar target/llama3-1.0.0-SNAPSHOT.jar \
  --model /path/to/model.gguf \
  --chat \
  --system-prompt "You are a helpful assistant that answers in Russian."
```

### Deterministic sampling with custom limits

```bash
java --add-modules jdk.incubator.vector \
  -jar target/llama3-1.0.0-SNAPSHOT.jar \
  --model /path/to/model.gguf \
  --prompt "List three JVM tuning tips." \
  --seed 42 \
  --temperature 0.2 \
  --top-p 0.9 \
  --max-tokens 200
```

## Use in another project

First, build and install the library to your local Maven repository:

```bash
mvn install
```

Then add it as a dependency in your project:

```xml
<dependency>
  <groupId>com.llama4j</groupId>
  <artifactId>llama3</artifactId>
  <version>1.0.0-SNAPSHOT</version>
</dependency>
```

Minimal Java usage example:

```java
import com.llama4j.model.Llama;
import com.llama4j.model.ModelLoader;
import com.llama4j.sampling.Sampler;
import com.llama4j.tokenizer.ChatFormat;

import java.nio.file.Path;
import java.util.List;

Path modelPath = Path.of("/path/to/model.gguf");
Llama model = ModelLoader.loadModel(modelPath, 4096, true);
Llama.State state = model.createNewState(16);

ChatFormat chatFormat = new ChatFormat(model.tokenizer());
List<Integer> promptTokens = chatFormat.encodeMessage(
    new ChatFormat.Message(ChatFormat.Role.USER, "Explain GGUF files in one sentence.")
);

List<Integer> responseTokens = Llama.generateTokens(
    model,
    state,
    0,
    promptTokens,
    chatFormat.getStopTokens(),
    512,
    Sampler.ARGMAX,
    false,
    null
);

String response = model.tokenizer().decode(responseTokens);
System.out.println(response);
```

## Run from source

```bash
mvn -q exec:java \
  -Dexec.mainClass=com.llama4j.cli.LlamaCli \
  -Dexec.args="--model /path/to/model.gguf --chat"
```
