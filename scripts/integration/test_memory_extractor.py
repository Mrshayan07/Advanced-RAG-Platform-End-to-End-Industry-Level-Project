
import asyncio

from app.services.memory_extractor import MemoryExtractor


async def main() -> None:
    extractor = MemoryExtractor()

    test_messages = [
        "My name is Shayan Javed.",
        "I study BS Computer Science at Sindh Madressatul Islam University.",
        "What is my name?",
        "I am an AI/ML engineer and I mainly work with Python.",
        "My password is Test123!",
    ]

    for message in test_messages:
        print("\n" + "=" * 70)
        print(f"USER: {message}")

        result = await extractor.extract(message)

        print("MEMORIES:")

        if not result.memories:
            print("  No durable memories found.")
            continue

        for memory in result.memories:
            print(
                f"  {memory.key} = {memory.value} "
                f"[{memory.memory_type}]"
            )


if __name__ == "__main__":
    asyncio.run(main())
