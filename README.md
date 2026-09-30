# CLI Q&A Agent - Gutermann Product Catalogue

A simple command-line tool that answers questions about a product
catalogue (product_overview.md). It only answers using what's actually
in the file - if the answer isn't there, it says so instead of guessing.

## How to run it

```bash
pip install -r requirements.txt
cp .env.example .env
python qa_agent.py
```

You need an LLM to generate answers. I ran a local model (llama3.1)
through Ollama, so it works fully offline with no API key needed. You
can also use OpenAI, Anthropic, or Groq instead if you'd rather use an
API - all of that is explained in .env.example.

​```
> What sensors does the AQUASCAN 760T use?
Agent: The AQUASCAN 760T uses True Sound Sensors (TSS).
> exit
​```

## How it works

1. Reads product_overview.md and splits it into chunks, one per product
2. Turns each chunk into numbers (embeddings) so it can compare meaning
3. When you ask a question, it finds the chunks closest to your question
4. Sends only those chunks to the LLM and tells it to answer using only
   that text - not its own general knowledge
5. If nothing relevant is found, it says "not enough information"
   instead of making something up

## Why I chunked it by product

Each product in the file has its own `###` heading with bullet points
under it, so it made sense to treat each product as one chunk instead of
splitting by a fixed number of words. That way a chunk never gets cut
off in the middle of a product's description.

One thing I had to fix: some sections (like "Permanent Leak Detection
Monitoring") have facts that apply to all the products under them, not
just one. So I made sure those shared facts get included in every
product's chunk from that section, otherwise a question about them
would miss the answer.

I also added the product name into the chunk text itself, not just as a
label. Without this, questions that mention a specific product name
(like comparing two models) weren't finding the right chunk properly.

## Things I tested

I tried all 7 sample questions from the assignment plus a few of my own.
Most worked correctly and gave grounded answers. The two "trick"
questions (about ordering a product that isn't released yet, and about
a feature that belongs to a different but similarly-named product) were
both answered correctly - it said "not enough info" instead of making
something up.

## What I'd improve with more time

- llama3.1 running locally is slow (a couple minutes per answer on my
  laptop). A hosted API would be much faster for real use.
- The current search uses word-matching + basic similarity. A better
  version would use a proper reranking step for tricky, multi-part
  questions.
- No memory between questions right now - each one is independent.
- Could add a proper test script instead of testing manually.