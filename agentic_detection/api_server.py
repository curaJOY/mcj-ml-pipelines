"""
FastAPI Production Server for Agentic Cyberbullying Detection
============================================================

Production-ready API endpoints for the CuraJOY challenge requirements:
- Real-time cyberbullying detection
- Agentic workflow explanation
- Challenge case validation
- Performance monitoring
"""

import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import traceback

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from agentic_cyberbullying_detector import AgenticCyberbullyingDetector, FinalDetectionResult
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CuraJOY Agentic Cyberbullying Detection API",
    description="Context-aware multi-agent cyberbullying detection with explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[AgenticCyberbullyingDetector] = None

# Request/Response Models
class DetectionRequest(BaseModel):
    """Request model for cyberbullying detection."""
    text: str = Field(..., min_length=1, max_length=Config.MAX_INPUT_LENGTH, description="Text to analyze")
    explain: bool = Field(True, description="Include detailed explanation in response")
    challenge_mode: bool = Field(True, description="Enable challenge-specific optimizations")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class AgentResultResponse(BaseModel):
    """Response model for individual agent results."""
    agent_name: str
    decision: str
    confidence: float
    reasoning: str
    evidence: List[str]
    processing_time: float


class DetectionResponse(BaseModel):
    """Response model for cyberbullying detection."""
    text: str
    is_cyberbullying: bool
    confidence: float
    explanation: str
    traditional_ml_score: float
    total_processing_time: float
    challenge_case_detected: Optional[str] = None
    agent_results: List[AgentResultResponse] = []
    timestamp: str
    api_version: str = "1.0.0"


class ChallengeTestResponse(BaseModel):
    """Response model for challenge test results."""
    sarcasm_case: Dict
    false_positive_case: Dict
    summary: Dict


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    detector_initialized: bool
    gemini_available: bool
    configuration: Dict


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the agentic detection system on startup."""
    global detector
    
    logger.info("🚀 Starting CuraJOY Agentic Cyberbullying Detection API")
    
    try:
        detector = AgenticCyberbullyingDetector()
        detector.initialize()
        logger.info("✅ Agentic detector initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        logger.error(traceback.format_exc())
        detector = None


# Main detection endpoint
@app.post("/detect", response_model=DetectionResponse)
async def detect_cyberbullying(request: DetectionRequest):
    """
    Detect cyberbullying using the agentic multi-agent system.
    
    This endpoint provides:
    - Multi-agent analysis with specialized reasoning
    - Context-aware sarcasm detection
    - False positive mitigation
    - Explainable AI decisions
    - Challenge case optimization
    """
    if not detector:
        raise HTTPException(
            status_code=503, 
            detail="Detection system not initialized. Please try again later."
        )
    
    try:
        # Perform detection
        start_time = time.time()
        result = detector.detect(request.text, explain=request.explain)
        
        # Convert agent results
        agent_responses = []
        for agent_result in result.agent_results:
            agent_responses.append(AgentResultResponse(
                agent_name=agent_result.agent_name,
                decision=agent_result.decision.value,
                confidence=agent_result.confidence,
                reasoning=agent_result.reasoning,
                evidence=agent_result.evidence[:3],  # Limit evidence for API response
                processing_time=agent_result.processing_time
            ))
        
        # Create response
        response = DetectionResponse(
            text=result.text,
            is_cyberbullying=result.is_cyberbullying,
            confidence=result.confidence,
            explanation=result.explanation,
            traditional_ml_score=result.traditional_ml_score,
            total_processing_time=result.total_processing_time,
            challenge_case_detected=result.challenge_case_detected,
            agent_results=agent_responses,
            timestamp=datetime.now().isoformat()
        )
        
        # Log the request for monitoring
        logger.info(f"Detection completed: {result.is_cyberbullying} (confidence: {result.confidence:.3f}, time: {result.total_processing_time:.3f}s)")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal detection error: {str(e)}"
        )


# Challenge validation endpoint
@app.post("/challenge/test", response_model=ChallengeTestResponse)
async def test_challenge_cases():
    """
    Test the system against specific CuraJOY challenge cases.
    
    Validates:
    - Sarcasm detection: "Hope you have a great day! 😊 (Just kidding, everyone will hate you)"
    - False positive prevention: "I'm literally dying of laughter at this meme!"
    """
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Detection system not initialized"
        )
    
    try:
        results = detector.test_challenge_examples()
        
        # Process results
        response_data = {}
        correct_count = 0
        total_count = 0
        
        for case_name, result in results.items():
            if result:
                is_correct = False
                if case_name == "sarcasm":
                    is_correct = result.is_cyberbullying
                elif case_name == "false_positive":
                    is_correct = not result.is_cyberbullying
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                response_data[f"{case_name}_case"] = {
                    "text": result.text,
                    "result": "CYBERBULLYING" if result.is_cyberbullying else "NOT_CYBERBULLYING",
                    "confidence": result.confidence,
                    "correct": is_correct,
                    "explanation": result.explanation,
                    "challenge_case_detected": result.challenge_case_detected,
                    "processing_time": result.total_processing_time
                }
        
        # Add summary
        response_data["summary"] = {
            "total_cases": total_count,
            "correct_cases": correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "challenge_requirements_met": correct_count == total_count
        }
        
        return ChallengeTestResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Challenge test error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Challenge test failed: {str(e)}"
        )


# Batch detection endpoint
@app.post("/detect/batch")
async def detect_batch(texts: List[str]):
    """
    Batch detection for multiple texts.
    
    Useful for:
    - Processing multiple messages
    - Performance testing
    - Bulk analysis
    """
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Detection system not initialized"
        )
    
    if len(texts) > 50:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 50 texts"
        )
    
    try:
        results = []
        start_time = time.time()
        
        for text in texts:
            if len(text.strip()) == 0:
                continue
                
            result = detector.detect(text, explain=False)
            results.append({
                "text": text,
                "is_cyberbullying": result.is_cyberbullying,
                "confidence": result.confidence,
                "challenge_case_detected": result.challenge_case_detected,
                "processing_time": result.total_processing_time
            })
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "summary": {
                "total_texts": len(results),
                "cyberbullying_detected": sum(1 for r in results if r["is_cyberbullying"]),
                "average_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0,
                "total_processing_time": total_time,
                "average_per_text": total_time / len(results) if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Batch detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring system status.
    """
    gemini_available = False
    if detector:
        try:
            # Check if Gemini is available
            context_agent = detector.agents.get('context_analysis')
            if context_agent and hasattr(context_agent, 'gemini_available'):
                gemini_available = context_agent.gemini_available
        except:
            pass
    
    return HealthResponse(
        status="healthy" if detector else "unhealthy",
        timestamp=datetime.now().isoformat(),
        detector_initialized=detector is not None,
        gemini_available=gemini_available,
        configuration={
            "gemini_model": Config.GEMINI_MODEL,
            "api_port": Config.API_PORT,
            "challenge_mode": Config.CHALLENGE_MODE,
            "max_input_length": Config.MAX_INPUT_LENGTH,
            "rate_limit": Config.RATE_LIMIT_PER_MINUTE
        }
    )


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """
    Get system metrics and performance data.
    """
    return {
        "api_version": "1.0.0",
        "system_status": "operational" if detector else "degraded",
        "configuration": {
            "gemini_model": Config.GEMINI_MODEL,
            "debug_mode": Config.DEBUG,
            "challenge_optimized": Config.CHALLENGE_MODE
        },
        "endpoints": {
            "detection": "/detect",
            "challenge_test": "/challenge/test",
            "batch_detection": "/detect/batch",
            "health": "/health"
        }
    }


# Documentation endpoint
@app.get("/")
async def root():
    """
    API documentation and quick start guide.
    """
    return {
        "message": "CuraJOY Agentic Cyberbullying Detection API",
        "version": "1.0.0",
        "description": "Multi-agent context-aware cyberbullying detection with explainable AI",
        "challenge_features": [
            "Sarcasm detection with masked malicious intent",
            "False positive reduction for friendly aggressive language",
            "Multi-agent reasoning with explanations",
            "Challenge case optimization"
        ],
        "quick_start": {
            "detection": "POST /detect with {'text': 'message to analyze'}",
            "challenge_test": "POST /challenge/test",
            "documentation": "GET /docs"
        },
        "examples": {
            "sarcasm": "Hope you have a great day! 😊 (Just kidding, everyone will hate you)",
            "false_positive": "I'm literally dying of laughter at this meme!"
        }
    }


def run_server():
    """Run the FastAPI server."""
    logger.info(f"🚀 Starting server on port {Config.API_PORT}")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=Config.API_PORT,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    run_server() 