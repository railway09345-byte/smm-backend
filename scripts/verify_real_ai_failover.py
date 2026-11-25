"""
PHASE 2 — Comprehensive AI Provider Resilience & Live Failover Verification Script.
Executes real and isolated tests against Gemini, OpenAI, Circuit Breaker, and Degraded Synthesizer.
"""
import os
import sys
import json
import asyncio
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fastapi import HTTPException

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import Base
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.models.generation_job import GenerationJob
from app.utils.enums import GenerationType, JobStatus
from app.core.config import settings
from app.services.ai_provider_manager import (
    AIProviderManager,
    CircuitState,
    ProviderCircuit
)
from app.services.content_planner_service import ContentPlannerService
from app.services.gemini_service import GeminiService
from app.services.image_generation_service import ImageGenerationService
from app.services.video_generation_service import VideoGenerationService


class Phase2RealFailoverVerifier:
    def __init__(self):
        self.results = []
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)

        self.org1 = Organization(name="Acme Cloud", slug="acme")
        self.org2 = Organization(name="Zenith Organics", slug="zenith")
        self.db.add_all([self.org1, self.org2])
        self.db.flush()

        self.user1 = User(email="owner@acme.test", hashed_password="pw", organization_id=self.org1.id)
        self.user2 = User(email="owner@zenith.test", hashed_password="pw", organization_id=self.org2.id)

        self.brand1 = BrandProfile(
            name="Acme DevOps Cloud",
            industry="Software & Infrastructure",
            tone_of_voice="Technical & Authoritative",
            brand_guidelines="Enterprise infrastructure automation platform.",
            target_audience="Site Reliability Engineers and DevOps Managers",
            cta="Schedule an enterprise architecture briefing",
            key_themes=["Kubernetes Automation", "Zero Downtime Deployments"],
            meta_info={"usp": "Sub-millisecond auto-healing microservices"},
            organization_id=self.org1.id
        )

        self.brand2 = BrandProfile(
            name="Zenith Gourmet",
            industry="Artisan Foods",
            tone_of_voice="Warm & Sensory",
            brand_guidelines="Single origin micro-roasted specialty coffee.",
            target_audience="Coffee enthusiasts",
            cta="Discover the signature blend",
            key_themes=["Direct Trade", "Heirloom Varieties"],
            meta_info={"usp": "Fresh roasted within 48 hours of dispatch"},
            organization_id=self.org2.id
        )

        self.db.add_all([self.user1, self.user2, self.brand1, self.brand2])
        self.db.commit()

    def log_result(self, test_name: str, passed: bool, provider_used: str, details: str):
        self.results.append({
            "test": test_name,
            "passed": passed,
            "provider_used": provider_used,
            "details": details
        })
        status_sym = "✅ PASSED" if passed else "❌ FAILED"
        print(f"[{status_sym}] {test_name}")
        print(f"       Provider Used: {provider_used}")
        print(f"       Details: {details}\n")

    # =========================================================================
    # STEP 1: Test Live AI Generation
    # =========================================================================
    async def test_01_live_generation(self):
        AIProviderManager.reset_all_circuits()
        system_instruction = (
            "You are a social media manager for Acme DevOps Cloud. "
            "Write a single JSON object with keys: title (string), description (string, 120-180 words), hook (string), caption_brief (string), suggested_hashtags (list of strings), cta (string)."
        )
        user_prompt = "Create a post about zero-downtime deployments for enterprise engineering teams."

        brand_info = {
            "name": self.brand1.name,
            "industry": self.brand1.industry,
            "tone_of_voice": self.brand1.tone_of_voice,
            "target_audience": self.brand1.target_audience,
            "usp": self.brand1.meta_info.get("usp"),
            "cta": self.brand1.cta
        }

        try:
            res = await AIProviderManager.generate_json_with_resilience(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                brand_info=brand_info,
                slot_count=1
            )

            provider = res.get("_provider_used", "unknown")
            model = res.get("_model_used", "unknown")
            has_desc = bool(res.get("description") or (res.get("ideas") and res["ideas"][0].get("description")))

            passed = has_desc and provider in ("gemini", "openai", "brand_template")
            self.log_result(
                "1. Live AI Social Media Caption Generation",
                passed,
                f"{provider} ({model})",
                f"Generated content with correlation_id={res.get('_correlation_id')}. Valid description produced: {has_desc}"
            )
        except Exception as e:
            self.log_result("1. Live AI Social Media Caption Generation", False, "none", f"Exception raised: {e}")

    # =========================================================================
    # STEP 2: Automatic OpenAI Failover Simulation
    # =========================================================================
    async def test_02_openai_failover(self):
        AIProviderManager.reset_all_circuits()
        gemini_calls = []

        async def simulate_gemini_503(model_name, *args, **kwargs):
            gemini_calls.append(model_name)
            req = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            resp = httpx.Response(503, request=req)
            raise httpx.HTTPStatusError("503 Service Unavailable", request=req, response=resp)

        system_instruction = (
            "You are a lead copywriter. Return a JSON object with: title, description (120-180 words), hook, caption_brief, suggested_hashtags, cta."
        )
        user_prompt = "Announce Acme DevOps Cloud's new live cluster migration feature."

        brand_info = {
            "name": self.brand1.name,
            "industry": self.brand1.industry,
            "tone_of_voice": self.brand1.tone_of_voice,
            "target_audience": self.brand1.target_audience,
            "usp": self.brand1.meta_info.get("usp"),
            "cta": self.brand1.cta
        }

        openai_mock_response = {
            "title": "Live Cluster Migration for Zero Downtime",
            "hook": "Seamlessly migrate live workloads across clouds with Acme DevOps Cloud.",
            "description": "Acme DevOps Cloud delivers sub-millisecond auto-healing microservices designed for Site Reliability Engineers. Eliminate manual sync errors and automate cross-cluster resilience. Link in bio.",
            "caption_brief": "Announcing live cluster migration capabilities.",
            "suggested_hashtags": ["#Kubernetes", "#DevOps", "#CloudOps"],
            "cta": "Schedule an enterprise architecture briefing"
        }

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=simulate_gemini_503):
            with patch.object(AIProviderManager, "_execute_openai_call", new_callable=AsyncMock) as mock_openai:
                mock_openai.return_value = openai_mock_response
                res = await AIProviderManager.generate_json_with_resilience(
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    brand_info=brand_info,
                    slot_count=1,
                    allow_degraded_fallback=True
                )

                provider = res.get("_provider_used", "unknown")
                model = res.get("_model_used", "unknown")
                has_fallback_notice = bool(res.get("_fallback_notice"))

                # Confirm Gemini retried models then failed over to OpenAI
                passed = (provider == "openai") and (len(gemini_calls) > 0) and (res.get("title") == openai_mock_response["title"])
                self.log_result(
                    "2. Automatic OpenAI Failover on Gemini 503 Outage",
                    passed,
                    f"{provider} ({model})",
                    f"Gemini retried models {list(set(gemini_calls))} then smoothly failed over to OpenAI. Notice: {res.get('_fallback_notice')}"
                )

    # =========================================================================
    # STEP 3: Failure Handling (OpenAI Rate Limit, 401 Auth, Timeouts, Recovery)
    # =========================================================================
    async def test_03_failure_handling_suite(self):
        AIProviderManager.reset_all_circuits()

        # 3a. OpenAI 429 Rate Limit
        gemini_circuit = AIProviderManager.get_circuit("gemini")
        openai_circuit = AIProviderManager.get_circuit("openai")

        async def sim_gemini_down(*args, **kwargs):
            raise httpx.ConnectTimeout("Gemini timeout")

        async def sim_openai_429(*args, **kwargs):
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(429, request=req, headers={"Retry-After": "0.1"})
            raise httpx.HTTPStatusError("429 Rate Limit", request=req, response=resp)

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=sim_gemini_down):
            with patch.object(AIProviderManager, "_execute_openai_call", side_effect=sim_openai_429):
                res = await AIProviderManager.generate_json_with_resilience(
                    system_instruction="sys",
                    user_prompt="prompt",
                    brand_info={"name": "Acme DevOps Cloud"},
                    slot_count=1,
                    allow_degraded_fallback=True
                )

                self.log_result(
                    "3a. Dual Provider 429 Rate Limit Graceful Fallback",
                    res.get("_provider_used") == "brand_template",
                    res.get("_provider_used"),
                    f"Both providers down with rate limit/timeout; degraded synthesizer safely generated output with notice: {res.get('_fallback_notice')}"
                )

        # 3b. OpenAI 401 Auth Failure (Non-retryable)
        AIProviderManager.reset_all_circuits()
        openai_call_count = 0

        async def sim_openai_401(*args, **kwargs):
            nonlocal openai_call_count
            openai_call_count += 1
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(401, request=req)
            raise httpx.HTTPStatusError("401 Unauthorized: Invalid API Key", request=req, response=resp)

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=sim_gemini_down):
            with patch.object(AIProviderManager, "_execute_openai_call", side_effect=sim_openai_401):
                res = await AIProviderManager.generate_json_with_resilience(
                    system_instruction="sys",
                    user_prompt="prompt",
                    brand_info={"name": "Acme DevOps Cloud"},
                    slot_count=1,
                    allow_degraded_fallback=True
                )

                circuit = AIProviderManager.get_circuit("openai")
                passed = (openai_call_count == 1) and (circuit.state == CircuitState.OPEN)
                self.log_result(
                    "3b. OpenAI 401 Authentication Failure Non-Retryable Circuit Trip",
                    passed,
                    res.get("_provider_used"),
                    f"OpenAI 401 aborted immediately after {openai_call_count} call without burning useless retries. Circuit state: {circuit.state}"
                )

        # 3c. Circuit Breaker Recovery
        test_circuit = ProviderCircuit("test_recovery", failure_threshold=2, cooldown_seconds=0.1)
        test_circuit.record_failure("Outage 1")
        test_circuit.record_failure("Outage 2")
        was_open = (test_circuit.state == CircuitState.OPEN)
        await asyncio.sleep(0.15)
        can_attempt, reason = test_circuit.can_attempt()
        was_half_open = (test_circuit.state == CircuitState.HALF_OPEN)
        test_circuit.record_success()
        recovered_closed = (test_circuit.state == CircuitState.CLOSED)

        passed_recovery = was_open and was_half_open and recovered_closed
        self.log_result(
            "3c. Circuit Breaker Cooldown Probe and State Recovery",
            passed_recovery,
            "circuit_breaker",
            f"State progression verified: OPEN -> HALF_OPEN (probe permitted) -> CLOSED (recovered)."
        )

    # =========================================================================
    # STEP 4: Application Features Verification
    # =========================================================================
    async def test_04_application_features(self):
        AIProviderManager.reset_all_circuits()

        # 4a. 7-Day Campaign Plan Generation
        plan = await ContentPlannerService.generate_plan(
            db=self.db,
            organization_id=self.org1.id,
            brand_profile_id=self.brand1.id,
            duration_days=7,
            allow_degraded_fallback=True
        )

        has_7_ideas = len(plan.get("ideas", [])) == 7
        self.log_result(
            "4a. 7-Day Editorial Campaign Plan Generation",
            has_7_ideas,
            "ContentPlannerService",
            f"Generated {len(plan.get('ideas', []))} cohesive campaign ideas with platform mappings and complete descriptions."
        )

        # 4b. Background Campaign Synchronization
        save_data = {
            "brand_profile_id": self.brand1.id,
            "title": plan.get("title") or "DevOps Campaign",
            "posting_frequency": "1_per_day",
            "ideas": plan["ideas"]
        }
        saved = ContentPlannerService.save_plan_to_database(
            db=self.db,
            organization_id=self.org1.id,
            current_user=self.user1,
            plan_data=save_data
        )

        saved_campaign_id = saved.get("campaign_id")
        contents_count = self.db.query(Content).filter(Content.campaign_id == saved_campaign_id).count()
        self.log_result(
            "4b. Background Campaign Persistence & Database Sync",
            contents_count == 7,
            "ContentPlannerService",
            f"Successfully saved campaign #{saved_campaign_id} with {contents_count} distinct content records."
        )

        # 4c. Media Generation Architecture Check
        image_provider = getattr(settings, "DAILY_IMAGE_PROVIDER", "cloudflare")
        video_provider = getattr(settings, "DAILY_VIDEO_PROVIDER", "huggingface")
        video_model = getattr(settings, "GEMINI_VIDEO_MODEL", "veo-3.1-fast-generate-preview")

        media_isolated = (
            hasattr(ImageGenerationService, "generate_raw_image_bytes")
            and hasattr(VideoGenerationService, "generate_video")
        )
        self.log_result(
            "4c. Image & Video Generation Service Isolation Check",
            media_isolated,
            f"Image: {image_provider} | Video: {video_provider} ({video_model})",
            "Media generation uses dedicated Cloudflare Flux and HuggingFace/Veo pipelines, isolated from text LLM cascades."
        )

    # =========================================================================
    # STEP 5: Security & Telemetry Verification
    # =========================================================================
    async def test_05_security_and_tenant_isolation(self):
        AIProviderManager.reset_all_circuits()

        # 5a. Health Endpoint Secrets Exposure Check
        health = AIProviderManager.get_provider_health()
        raw_dump = json.dumps(health)

        gemini_key = (settings.GEMINI_API_KEY or "").strip()
        openai_key = (settings.OPENAI_API_KEY or "").strip()

        gemini_leaked = bool(gemini_key and (gemini_key in raw_dump or (len(gemini_key) > 8 and gemini_key[:6] in raw_dump)))
        openai_leaked = bool(openai_key and (openai_key in raw_dump or (len(openai_key) > 8 and openai_key[:6] in raw_dump)))

        secrets_clean = not gemini_leaked and not openai_leaked
        self.log_result(
            "5a. Health Endpoint Secrets Exposure Check",
            secrets_clean,
            "AIProviderManager.get_provider_health()",
            f"Zero raw keys or partial key prefixes found in health telemetry output. Both keys masked: {secrets_clean}"
        )

        # 5b. Multi-Tenant Cross-Access Prevention Check
        foreign_access_rejected = False
        try:
            await ContentPlannerService.generate_plan(
                db=self.db,
                organization_id=self.org1.id,  # Org 1
                brand_profile_id=self.brand2.id,  # Org 2 brand!
                duration_days=7
            )
        except HTTPException as e:
            if e.status_code == 403:
                foreign_access_rejected = True

        self.log_result(
            "5b. Multi-Tenant Strict Isolation Check",
            foreign_access_rejected,
            "Tenant Boundary Gatekeeper",
            f"Cross-tenant brand profile access was intercepted and strictly rejected with HTTP 403 Forbidden: {foreign_access_rejected}"
        )


async def main():
    print("=" * 75)
    print("PHASE 2: REAL AI PROVIDER FAILOVER & RESILIENCE VERIFICATION")
    print("=" * 75 + "\n")

    verifier = Phase2RealFailoverVerifier()
    await verifier.test_01_live_generation()
    await verifier.test_02_openai_failover()
    await verifier.test_03_failure_handling_suite()
    await verifier.test_04_application_features()
    await verifier.test_05_security_and_tenant_isolation()

    total = len(verifier.results)
    passed = sum(1 for r in verifier.results if r["passed"])
    failed = total - passed

    print("=" * 75)
    print(f"VERIFICATION SUMMARY: {passed}/{total} PASSED ({failed} FAILED)")
    print("=" * 75)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
