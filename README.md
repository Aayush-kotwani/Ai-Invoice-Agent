
# Invoice Action Agent (A2A 1.0)

FastAPI starter for the invoice-agent task: public Agent Card, bearer authentication, A2A HTTP+JSON routes, AI-assisted evidence-cited proposals, SQLite task persistence, message idempotency, proposal/result binding, and receipt artifacts.

## Important limitations before submitting

This is a working starter, not a guarantee of full grader compliance. You must test it against the live contract and hidden audit cases. The task's exact source package shape and policy may require adjusting the decision prompt and validation. Render's free filesystem is ephemeral; use a persistent disk or managed database for durable task state. The included `render.yaml` is a starting point and the `/var/data` path requires a persistent disk configured in Render.

## Local setup

1. Python 3.11+ recommended.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`, then set `A2A_BEARER_TOKEN` and `OPENAI_API_KEY`. Export the variables into your shell (or use your preferred environment loader).
5. Set `A2A_BASE_URL` to the public URL you will submit. Locally, it can be `http://localhost:8000/a2a` for smoke testing, but the grader requires public HTTPS.
6. Run:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

Check the public card at `GET /.well-known/agent-card.json`. Authenticated routes require:
- `Authorization: Bearer <A2A_BEARER_TOKEN>`
- `A2A-Version: 1.0`
- `Content-Type: application/a2a+json`

## Deploy to Render

1. Push this folder to a GitHub repository.
2. In Render, create a Web Service from the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `A2A_BEARER_TOKEN`: long random secret (Render can generate one)
   - `A2A_BASE_URL`: `https://YOUR-SERVICE.onrender.com/a2a`
   - `OPENAI_API_KEY`: your provider key
   - `OPENAI_BASE_URL`: `https://aipipe.org/openrouter/v1` (or your provider's compatible endpoint)
   - `OPENAI_MODEL`: `google/gemini-2.5-flash-lite`
   - `DATABASE_PATH`: `/var/data/invoice_agent.sqlite3`
6. Configure a persistent disk mounted at `/var/data` if the plan supports it. If it does not, use a managed persistent database or a host with persistent storage.
7. After deployment, verify:
   - `https://YOUR-SERVICE.onrender.com/` returns health JSON.
   - `https://YOUR-SERVICE.onrender.com/.well-known/agent-card.json` returns the Agent Card.
   - `supportedInterfaces[0].url` exactly matches the base URL you submit.
8. Submit `https://YOUR-SERVICE.onrender.com/a2a` as the A2A interface base URL.

## API route overview

- `GET /.well-known/agent-card.json` — public Agent Card.
- `POST /a2a/message:send` — initial batch or results continuation.
- `GET /a2a/tasks/{task_id}` — read one owned task.
- `GET /a2a/tasks` — list tasks belonging to the bearer principal.
- `POST /a2a/tasks/{task_id}:cancel` — cancel a nonterminal task.

## Smoke-test headers

```http
Authorization: Bearer YOUR_SECRET
A2A-Version: 1.0
Content-Type: application/a2a+json
```

The implementation fails closed if `A2A_BEARER_TOKEN` is missing. Do not commit `.env`, API keys, real invoices, or payment credentials.
