from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from agentweave.scheduler.engine import (
    submit_request_rag,
    submit_request_rpg,
    submit_request_kb_rag,
    submit_request_deep_search,
    get_request_state,
)

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    doc: str = ""
    config: Optional[Dict[str, Any]] = None
    # agent_type
    agent_type: Optional[str] = None


class ChatResponse(BaseModel):
    request_id: str
    message: str


# class ResultResponse(BaseModel):
#     request_id: str
#     status: str
#     current_round: int
#     total_rounds: int


@router.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        if req.agent_type == "rag":
            if not req.doc.strip():
                raise HTTPException(status_code=400, detail="rag requires a non-empty doc")
            req_id = await submit_request_rag(req.query, req.doc, req.config)
        elif req.agent_type == "rpg":
            req_id = await submit_request_rpg(req.query, req.config)
        elif req.agent_type == "kb_rag":
            req_id = await submit_request_kb_rag(req.query, req.config)
        elif req.agent_type == "deep_search":
            req_id = await submit_request_deep_search(req.query, req.config)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown agent_type: {req.agent_type}"
            )
        return ChatResponse(request_id=req_id, message="Request accepted.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/v1/result/{request_id}", response_model=ResultResponse)
# async def get_result_endpoint(request_id: str):
#     """
#     """
#     state = get_request_state(request_id)
#     if not state:
#         raise HTTPException(status_code=404, detail="Request ID not found.")
#
#     return ResultResponse(
#         request_id=state.req_id,
#         status=state.status,
#         current_round=state.current_round_idx,
#         total_rounds=state.current_round_idx,
#         answers=state.round_answers,
#     )
