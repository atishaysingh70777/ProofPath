PROOFPATH AI — BEGINNER SETUP AND CODE NOTES
============================================

WHAT THIS VERSION DOES
----------------------
1. Searches Wikipedia for pages related to the question.
2. Retrieves relevant Wikipedia text passages.
3. Ranks passages using text similarity (this is a relevance score, not a truth score).
4. If an OpenAI API key is configured, asks an AI model to assess the question using those passages.
5. Shows the AI assessment, evidence passages, and links so the user can verify them.
6. Provides Britannica and BBC as manual reference links.
7. The Hindu is removed.

IMPORTANT ABOUT BRITANNICA AND BBC
----------------------------------
This prototype does not automatically scrape Britannica or BBC page text. Their terms restrict automated extraction and AI analysis without permission. The app displays links to those sites so users can open and verify relevant information themselves. Get appropriate permission before adding automated content retrieval from those sites.

REQUIREMENTS
------------
Python 3.10 or newer recommended.

INSTALL
-------
Open PowerShell in the folder containing app.py, then run:

python -m pip install streamlit requests beautifulsoup4 scikit-learn

ENABLE AI
---------
The AI part needs an OpenAI API key. ChatGPT subscription and API billing are separate; API usage may cost money.

In PowerShell, for the current terminal session only, run:

$env:OPENAI_API_KEY = "paste-your-own-api-key-here"

Optional model setting (default is gpt-4o-mini):
$env:PROOFPATH_AI_MODEL = "gpt-4o-mini"

Then start the app in that same PowerShell window:

python -m streamlit run app.py

If you do not set OPENAI_API_KEY, the evidence search still works, but AI assessment will be disabled.
Never paste your API key into app.py or share it in screenshots, GitHub, or with other people.

RUN THE APP
-----------
python -m streamlit run app.py

CODE NOTES — SIMPLE EXPLANATION
-------------------------------
- DEFAULT_SOURCES: initial source list used by the app.
- search_wikipedia(): asks Wikipedia's public API for matching pages.
- fetch_wikipedia_extract(): downloads the plain-text extract of a Wikipedia page.
- retrieve_research(): gathers pages and records which sources were available.
- split_sentences(): splits page text into smaller sentences.
- rank_evidence(): sorts sentences by how closely their words match the question.
- ai_fact_check(): sends the question and selected evidence to the AI API and asks for a cautious assessment.
- st.session_state: keeps the last results visible while Streamlit redraws the page.
- REFERENCE_LINKS: the Britannica and BBC links shown for manual verification.

HOW TO TEST
-----------
Try questions such as:
- Who was Aryabhata and what did he contribute?
- What causes climate change?
- Who invented the telephone?
- Why do earthquakes happen?
- How does artificial intelligence work?

LIMITATIONS
-----------
- This is a student prototype, not a professional fact-checking service.
- AI can be wrong. Always open the sources and verify important claims.
- Search coverage currently relies mainly on Wikipedia, so some questions may return weak or no evidence.
- It does not guarantee up-to-the-minute news coverage.
- An OpenAI API key and available API quota are needed for AI-generated assessments.
