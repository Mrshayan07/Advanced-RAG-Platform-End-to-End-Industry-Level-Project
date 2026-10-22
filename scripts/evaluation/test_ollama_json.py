from app.evaluation.models.ollama import OllamaJudge

judge = OllamaJudge()

prompt = """
Return ONLY valid JSON.
Do not use markdown.

Output exactly this structure:

{
  "claims": [
    {
      "claim": "The answer is supported by the context.",
      "truth": true
    }
  ],
  "reason": "The context supports the answer."
}

Context:
Shayan Javed is a Full Stack Developer and MERN Stack Developer.

Answer:
Shayan Javed is a Full Stack Developer specializing in MERN-stack development.
"""

result = judge.generate(prompt)

print("RAW OUTPUT:")
print(repr(result))

print("\nNORMAL OUTPUT:")
print(result)