import asyncio
import logging
from contextlib import asynccontextmanager
from agentweave.agents.kb_rag import KnowledgeBaseRAGAgent
from agentweave.agents.rag import RAGAgent
import uvicorn
from fastapi import FastAPI

from agentweave.utils.config import app_config
from agentweave.agents.rpg import RPGAgent
from agentweave.agents.deep_search import DeepSearchAgent
from agentweave.scheduler.engine import (
    init_scheduler,
    run_scheduler_loop,
    analyze_performance,
)
from agentweave.server.api import router as api_router
from agentweave.utils.logger import force_setup_logging


force_setup_logging()
logger = logging.getLogger("Main")

agent_class = {
    "rag": RAGAgent,
    "kb_rag": KnowledgeBaseRAGAgent,
    "rpg": RPGAgent,
    "deep_search": DeepSearchAgent,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    args = app_config.args

    if args["agent_type"] == "rag":
        from agentweave.core.modules_rag import ModelRegistry

        ModelRegistry.load_all()
    elif args["agent_type"] == "kb_rag":
        from agentweave.core.modules_kb_rag import ModelRegistry

        ModelRegistry.load_all()
    elif args["agent_type"] == "rpg":
        from agentweave.core.modules_rpg import ModelRegistry

        ModelRegistry.load_all()
    elif args["agent_type"] == "deep_search":
        from agentweave.core.modules_deep_search import ModelRegistry

        ModelRegistry.load_all()
    else:
        raise ValueError("not support ")

    agent = agent_class[args["agent_type"]]()

    init_scheduler(agent)
    worker_task = asyncio.create_task(run_scheduler_loop())

    yield

    logger.info("Server shutting down...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    try:
        analyze_performance()
    finally:
        ModelRegistry.shutdown_all()


app = FastAPI(lifespan=lifespan, title="AgentWeave Serving")
app.include_router(api_router)

if __name__ == "__main__":
    args = app_config.args
    print(args)
    uvicorn.run(app, access_log=False, host="0.0.0.0", port=30000)
