# AgentWeave

This is the demo of AgentWeave.

```bash
git clone https://github.com/Guodanding/agentweave.git
cd agentweave
bash ./install.sh
```
Create your local configuration, then edit `.env`:

```bash
cp .env.example .env
```

To run KB-RAG, build the knowledge base:

```bash
cd kb
bash build_kb.sh
```

Start the server:

```bash
bash ./script/run_server.sh
```

Supported agent types are `rag`, `kb_rag`, `rpg`, and `deep_search`. Start the server
with the matching `AGENT_TYPE` in `.env`.

Run client request:

```bash
bash ./script/run_client.sh rag
bash ./script/run_client.sh kb_rag
bash ./script/run_client.sh rpg
bash ./script/run_client.sh deep_search
```

Stop the Server:
```bash
Ctrl-C
```

Read the answers or analyze the performance:
```bash
./script/performance
```