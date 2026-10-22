from deepeval.models import OllamaModel


def main():
    print("=" * 60)
    print("DEEPEVAL + OLLAMA SMOKE TEST")
    print("=" * 60)

    print("\nCreating OllamaModel...")

    model = OllamaModel(
        model="qwen3:4b-instruct",
        base_url="http://localhost:11434",
        temperature=0,
    )

    print("Model created successfully.")
    print("Sending test prompt to Qwen3...\n")

    response = model.generate(
        "Reply with exactly this text and nothing else: LOCAL QWEN WORKING"
    )

    print("Qwen3 response:")
    print(response)

    print("\n" + "=" * 60)

    if "LOCAL QWEN WORKING" in str(response):
        print("PASS: DeepEval -> Ollama -> Qwen3 is working.")
    else:
        print("FAIL: Unexpected response from Qwen3.")

    print("=" * 60)


if __name__ == "__main__":
    main()
