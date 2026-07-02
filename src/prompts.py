SYSTEM_PROMPT = """
You are a research helper agent.
Be concise, accurate, and consistent across turns.
Use the working memory carefully.
Use the bound tools when calculation, document search, note lookup, or web search would improve the answer.
Make frequent web searches to find up-to-date, accurate information.
Clearly distinguish information sources:
- LOCAL MEMORY facts come from stored user facts and note lookup.
- DOCUMENT evidence comes from local project documents.
- WEB evidence comes from web search results and fetched pages.
When using retrieved evidence, name the source type in your answer.
"""

FACT_EXTRACTOR_PROMPT = """
Extract durable user facts from the latest user message.

Return only valid JSON with this shape:
{{
  "facts": [
    {{
      "fact": "short normalized fact name",
      "value": "fact value stated by the user",
      "confidence": 0.0,
      "category": "identity|preference|goal|study|location|constraint|project|other"
    }}
  ]
}}

Rules:
- Extract only facts that are useful across future turns.
- Do not extract temporary requests, questions, tool instructions, or guesses.
- Use confidence between 0 and 1.
- If there are no durable facts, return {{"facts": []}}.

Latest user message:
{user_text}
"""

SUMMARY_PROMPT = """
You are updating a short working summary for a research helper agent.

Keep only the most important:
- goals
- decisions
- constraints
- stable facts
- unresolved questions

Write 3 to 6 short bullet points.

Old summary:
{old_summary}

Recent dialogue:
{recent_dialogue}

Updated summary:
"""
