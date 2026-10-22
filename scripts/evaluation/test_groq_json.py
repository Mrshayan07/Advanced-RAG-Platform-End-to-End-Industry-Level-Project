import json

from app.evaluation.models.groq import GroqJudge


judge = GroqJudge()

prompt = """
Return ONLY valid JSON.
Do not use markdown.

Output exactly this structure:

{
  "score": 1,
  "reason": "The answer is supported by the context."
}
"""

result = judge.generate(prompt)

print("RAW OUTPUT:")
print(repr(result))

print("\nNORMAL OUTPUT:")
print(result)

print("\nJSON TEST:")

try:
    parsed = json.loads(result)
    print("VALID JSON")
    print(parsed)
except json.JSONDecodeError as exc:
    print("INVALID JSON")
    print(exc)