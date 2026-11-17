# georgian-legal-rag

RAG over Georgian laws, running locally with Ollama. I wanted to know how much retrieval actually helps small open models in Georgian, so most of the work here went into the evaluation, not the chatbot.

Everything runs on my laptop (RTX 4070, 8 GB VRAM).

## Results

I wrote 34 test questions the way a normal person would ask them (`eval/questions.csv`). 30 can be answered from the laws, 4 can't (minimum wage, income tax, driving age, military service). For those 4 the right answer is "I don't know".

Same questions, same judge, with and without retrieval:

| Model | No RAG | RAG | Refused out-of-scope (RAG) | Cited the right article | Time per answer (RAG) |
|---|---|---|---|---|---|
| gemma3:12b | 41% | 82% | 4/4 | 77% | 21 s |
| qwen3:8b (thinking off) | 12% | 71% | 4/4 | 83% | 25 s |
| gemma3:4b | 6% | 53% | 3/4 | 40% | 9 s |

Retrieval found the correct article for 29 of the 30 answerable questions in all three runs. So the gap between the models is not about search, it's about how well each model reads the Georgian text it gets.

gemma3:12b by question type:

| | fact | condition | not in docs |
|---|---|---|---|
| RAG | 79% | 83% | 100% |
| No RAG | 50% | 33% | 0% |

### Things I noticed

Without documents the models make things up with full confidence. gemma3:12b told me the President is elected by the people for 6 years (it's the electoral college, 5 years), that the electoral college has 380 members (300), that late salary is compensated at 0.5% per day (0.07%), and that citizenship can be taken away (the Constitution says it can't). For the 4 out-of-scope questions it invented an answer every time, e.g. "the minimum wage is 800 lari". With RAG it said it couldn't find the answer in all 4 cases. For me that's the most useful result.

RAG doesn't matter much for well-known facts. The model already knew there are 150 MPs and that the Prime Minister leads the government. Where retrieval helped was exact numbers, units (24 *working* days of leave, not calendar days), conditions, and saying "I don't know".

qwen3 has trouble with long Georgian context. On the consumer and data protection questions it often ignored the article it was given. It said a child can consent to data processing from age 12, while the retrieved article says 16. It also said a data breach must be reported within 10 working days, but the article in its context says 72 hours.

### Where it still fails

- Missed retrieval once: the question about the deadline for complaining to the consumer agency (2 years, article 31) didn't retrieve article 31.
- Ignored a condition: "I bought something in a shop, can I return it within 14 days?" All models answered yes. The 14-day right is only for online and off-premises purchases.
- Wrong citations: about 1 in 4 answers cite a source number that exists but isn't the article the fact came from.

### About the scores

The answers are graded by gemma3:12b against my answer key, so the numbers aren't perfect. I went through all gemma3:12b answers by hand and disagreed with the judge on about 3 of 68. For qwen3 it marked two correct answers as wrong (q21, q22). My own count gives roughly 85-90% with RAG and 35% without for gemma3:12b, so the conclusion is the same.

## How it works

1. `src/ingest.py` reads the laws (Word files from matsne.gov.ge) and splits them into articles. Long articles get split into parts.
2. `src/index.py` embeds the chunks with bge-m3 and stores them in ChromaDB.
3. `src/search.py` finds the closest chunks to a question.
4. `src/rag.py` takes the whole articles those chunks belong to, gives them to the LLM, and asks it to answer only from them and cite sources as [1], [2].
5. `src/evaluate.py` runs the question set and writes the tables above.

A few things I changed after testing:

- **Whole articles instead of chunks.** In the first version the model got the middle part of a long article and answered a side case instead of the main rule. Now search still works on small chunks, but the model gets the full article.
- **Citations by number.** When I asked the model to cite article numbers directly, it cited articles it had never seen. Now it cites [1], [2], the code maps them to real articles, and any number that doesn't exist gets flagged.
- **Choosing the model.** Before building anything I asked gemma3:4b, qwen3:8b, llama3.1:8b and gemma3:12b the same question in Georgian. Only gemma3:12b answered without mixing in other scripts (the 4B model put Thai, Bengali and Chinese characters in the middle of Georgian sentences).

## Data

Four laws, current versions from matsne.gov.ge: the Constitution, the Labour Code, the Law on Consumer Rights Protection and the Law on Personal Data Protection. That's 294 articles, 458 chunks. The files aren't in the repo; download them as .docx into `data/docs/`.

## Running it

```bash
pip install -r requirements.txt
ollama pull bge-m3
ollama pull gemma3:12b

python src/ingest.py
python src/index.py
python src/rag.py "რამდენი დღე ანაზღაურებადი შვებულება მეკუთვნის წელიწადში?"
python src/rag.py "..." --no_rag

python src/evaluate.py
python src/evaluate.py --llms gemma3:4b qwen3:8b
```

On an 8 GB GPU I had Ollama crash while switching between models. Setting `OLLAMA_MAX_LOADED_MODELS=1` fixed it.

## Todo

- Hybrid search (BM25 + embeddings) for the missed article
- Make the model check conditions before saying yes
- Check that each cited article actually contains the claim
- Simple web UI

Not legal advice, just an experiment.
