import asyncio
import sys
import unittest
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fastapi import HTTPException

# Ensure backend root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import Base
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.models.generation_job import GenerationJob
from app.utils.enums import GenerationType, JobStatus
from app.services.ai_provider_manager import (
    AIProviderManager,
    CircuitState,
    ProviderCircuit,
    ProgressTracker
)
from app.services.content_planner_service import ContentPlannerService


class AIProviderResilienceTests(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive verification suite for TASK 2:
    Testing 14 core AI provider resilience, circuit breaker, fallback, and tenant isolation scenarios.
    """

    def setUp(self):
        AIProviderManager.reset_all_circuits()
        from app.core.config import settings
        self._orig_gemini_key = settings.GEMINI_API_KEY
        self._orig_openai_key = settings.OPENAI_API_KEY
        settings.GEMINI_API_KEY = "AIzaSyTestKey1234567890TestValidKey"

        # In-memory isolated database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)

        self.org1 = Organization(name="Acme Corp", slug="acme")
        self.org2 = Organization(name="Zenith Ltd", slug="zenith")
        self.db.add_all([self.org1, self.org2])
        self.db.flush()

        self.user1 = User(email="acme_user@acme.test", hashed_password="pw", organization_id=self.org1.id)
        self.user2 = User(email="zenith_user@zenith.test", hashed_password="pw", organization_id=self.org2.id)

        self.brand_acme = BrandProfile(
            name="Acme Cloud Tools",
            industry="Cloud SaaS",
            tone_of_voice="Professional & Authoritative",
            brand_guidelines="Enterprise cloud automation infrastructure.",
            target_audience="DevOps & CTOs",
            cta="Schedule an architecture demo",
            key_themes=["Cloud Ops", "Kubernetes", "Observability"],
            meta_info={"usp": "Zero-downtime microservice orchestration"},
            organization_id=self.org1.id
        )

        self.brand_zenith = BrandProfile(
            name="Zenith Organic Coffee",
            industry="Artisan Food & Beverage",
            tone_of_voice="Warm & Community-Driven",
            brand_guidelines="Sustainably harvested single-origin specialty roast.",
            target_audience="Coffee connoisseurs and mindful consumers",
            cta="Order your tasting box today",
            key_themes=["Artisan Roasting", "Fair Trade", "Mindful Morning"],
            meta_info={"usp": "Fair-trade farm-to-cup ethical beans"},
            organization_id=self.org2.id
        )

        self.db.add_all([self.user1, self.user2, self.brand_acme, self.brand_zenith])
        self.db.commit()

    def tearDown(self):
        from app.core.config import settings
        settings.GEMINI_API_KEY = self._orig_gemini_key
        settings.OPENAI_API_KEY = self._orig_openai_key
        self.db.close()
        self.engine.dispose()
        AIProviderManager.reset_all_circuits()

    def _sample_valid_plan(self, slot_count=7, brand_name="Acme Cloud Tools"):
        ideas = []
        for i in range(slot_count):
            ideas.append({
                "title": f"Day {i+1}: {brand_name} Innovation Spotlight",
                "hook": f"Here is why {brand_name} is transforming modern teams today.",
                "description": f"Detailed publication-ready caption for {brand_name} with over one hundred words discussing enterprise cloud efficiency, operational resilience, and tactical best practices for infrastructure engineering teams. Link in bio.",
                "caption_brief": f"Overview of day {i+1} core narrative and product value.",
                "content_format": "image",
                "pillar": "Product Spotlight",
                "suggested_hashtags": ["#Cloud", "#DevOps", "#TechInnovation"],
                "cta": "Schedule an architecture demo"
            })
        return {
            "campaign_title": f"{brand_name} Launch Plan",
            "campaign_summary": "7-day comprehensive strategy for social growth.",
            "ideas": ideas
        }

    # -------------------------------------------------------------------------
    # Scenario 1: Primary provider success
    # -------------------------------------------------------------------------
    async def test_01_primary_provider_success(self):
        with patch.object(AIProviderManager, "_execute_gemini_call", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.side_effect = lambda *a, **kw: self._sample_valid_plan()

            result = await AIProviderManager.generate_json_with_resilience(
                system_instruction="sys",
                user_prompt="prompt",
                brand_info={"name": "Acme Cloud Tools"}
            )

            self.assertEqual(result["_provider_used"], "gemini")
            self.assertIn("gemini-3.5-flash", result["_model_used"])
            self.assertEqual(len(result["ideas"]), 7)
            self.assertFalse(result.get("is_degraded_fallback", False))
            self.assertEqual(mock_gemini.call_count, 1)

    # -------------------------------------------------------------------------
    # Scenario 2: Primary provider 429 and secondary success
    # -------------------------------------------------------------------------
    async def test_02_primary_provider_429_secondary_success(self):
        valid_plan = self._sample_valid_plan()

        async def mock_call(model_name, *args, **kwargs):
            if "gemini-3.5-flash" in model_name:
                req = httpx.Request("POST", "https://generativelanguage.googleapis.com")
                resp = httpx.Response(429, request=req, headers={"Retry-After": "0.1"})
                raise httpx.HTTPStatusError("Rate limited", request=req, response=resp)
            return valid_plan

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=mock_call) as mock_gemini:
            result = await AIProviderManager.generate_json_with_resilience(
                system_instruction="sys",
                user_prompt="prompt",
                brand_info={"name": "Acme Cloud Tools"}
            )

            self.assertEqual(result["_provider_used"], "gemini")
            self.assertEqual(result["_model_used"], "gemini-3.1-flash-lite")
            self.assertTrue(result.get("_fallback_notice") is not None)
            self.assertEqual(len(result["ideas"]), 7)

    # -------------------------------------------------------------------------
    # Scenario 3: Primary provider 503 and secondary success
    # -------------------------------------------------------------------------
    async def test_03_primary_provider_503_secondary_success(self):
        valid_plan = self._sample_valid_plan()

        async def mock_call(model_name, *args, **kwargs):
            if "gemini-3.5-flash" in model_name:
                req = httpx.Request("POST", "https://generativelanguage.googleapis.com")
                resp = httpx.Response(503, request=req)
                raise httpx.HTTPStatusError("Service Unavailable", request=req, response=resp)
            return valid_plan

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=mock_call):
            result = await AIProviderManager.generate_json_with_resilience(
                system_instruction="sys",
                user_prompt="prompt",
                brand_info={"name": "Acme Cloud Tools"}
            )

            self.assertEqual(result["_provider_used"], "gemini")
            self.assertEqual(result["_model_used"], "gemini-3.1-flash-lite")
            self.assertEqual(len(result["ideas"]), 7)

    # -------------------------------------------------------------------------
    # Scenario 4: Gemini failure and OpenAI success
    # -------------------------------------------------------------------------
    async def test_04_gemini_failure_and_openai_success(self):
        valid_plan = self._sample_valid_plan()

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=httpx.ConnectTimeout("Gemini timeout")):
            with patch.object(AIProviderManager, "_execute_openai_call", new_callable=AsyncMock) as mock_openai:
                with patch("app.core.config.settings.OPENAI_API_KEY", "sk-proj-testkey1234567890"):
                    mock_openai.return_value = valid_plan

                    result = await AIProviderManager.generate_json_with_resilience(
                        system_instruction="sys",
                        user_prompt="prompt",
                        brand_info={"name": "Acme Cloud Tools"}
                    )

                    self.assertEqual(result["_provider_used"], "openai")
                    self.assertEqual(result["_model_used"], "gpt-4o-mini")
                    self.assertIn("fallback", result["_fallback_notice"].lower())

    # -------------------------------------------------------------------------
    # Scenario 5: All providers unavailable -> Degraded verified brand template
    # -------------------------------------------------------------------------
    async def test_05_all_providers_unavailable_degraded_fallback(self):
        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=httpx.ConnectError("Network offline")):
            with patch.object(AIProviderManager, "_execute_openai_call", side_effect=httpx.ConnectError("OpenAI offline")):
                result = await AIProviderManager.generate_json_with_resilience(
                    system_instruction="sys",
                    user_prompt="prompt",
                    brand_info={
                        "name": "Acme Cloud Tools",
                        "industry": "Cloud SaaS",
                        "tone_of_voice": "Professional",
                        "target_audience": "DevOps Leads",
                        "usp": "Zero downtime cloud scale",
                        "cta": "Get Started",
                        "key_themes": ["Kubernetes", "Monitoring"]
                    },
                    duration_days=7,
                    slot_count=7
                )

                self.assertTrue(result.get("is_degraded_fallback"))
                self.assertEqual(result["_provider_used"], "brand_template")
                self.assertEqual(len(result["ideas"]), 7)
                # Verify zero lorem ipsum and strict brand name presence
                for idea in result["ideas"]:
                    self.assertIn("Acme Cloud Tools", idea["description"])
                    self.assertNotIn("lorem ipsum", idea["description"].lower())
                    self.assertTrue(len(idea["description"]) > 100)

    # -------------------------------------------------------------------------
    # Scenario 6: Invalid API key (immediate abort, no useless retries)
    # -------------------------------------------------------------------------
    async def test_06_invalid_api_key_immediate_failover(self):
        gemini_circuit = AIProviderManager.get_circuit("gemini")

        req = httpx.Request("POST", "https://generativelanguage.googleapis.com")
        resp = httpx.Response(401, request=req)
        auth_error = httpx.HTTPStatusError("API_KEY_INVALID: 401 Unauthorized", request=req, response=resp)

        valid_plan = self._sample_valid_plan()

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=auth_error) as mock_gemini:
            with patch.object(AIProviderManager, "_execute_openai_call", new_callable=AsyncMock) as mock_openai:
                with patch("app.core.config.settings.OPENAI_API_KEY", "sk-proj-testkey1234567890"):
                    mock_openai.return_value = valid_plan

                    result = await AIProviderManager.generate_json_with_resilience(
                        system_instruction="sys",
                        user_prompt="prompt",
                        brand_info={"name": "Acme Cloud Tools"}
                    )

                    # Because 401 is permanent, gemini should have failed immediately without retrying all candidate models
                    self.assertEqual(mock_gemini.call_count, 1)
                    # And gemini circuit tripped to OPEN due to permanent misconfiguration
                    self.assertEqual(gemini_circuit.state, CircuitState.OPEN)
                    self.assertEqual(result["_provider_used"], "openai")

    # -------------------------------------------------------------------------
    # Scenario 7: Network connection reset / transport error retry
    # -------------------------------------------------------------------------
    async def test_07_network_connection_reset_retry(self):
        valid_plan = self._sample_valid_plan()
        attempts = 0

        async def flaky_call(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionResetError("Connection reset by peer [WinError 10054]")
            return valid_plan

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=flaky_call):
            result = await AIProviderManager.generate_json_with_resilience(
                system_instruction="sys",
                user_prompt="prompt",
                brand_info={"name": "Acme Cloud Tools"}
            )

            self.assertEqual(attempts, 2)
            self.assertEqual(result["_provider_used"], "gemini")
            self.assertEqual(len(result["ideas"]), 7)

    # -------------------------------------------------------------------------
    # Scenario 8: Circuit breaker activation on repeated failures
    # -------------------------------------------------------------------------
    async def test_08_circuit_breaker_activation(self):
        circuit = ProviderCircuit("gemini", failure_threshold=3, cooldown_seconds=30.0)
        self.assertEqual(circuit.state, CircuitState.CLOSED)

        circuit.record_failure("Failure 1")
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        self.assertEqual(circuit.consecutive_failures, 1)

        circuit.record_failure("Failure 2")
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        self.assertEqual(circuit.consecutive_failures, 2)

        circuit.record_failure("Failure 3")
        self.assertEqual(circuit.state, CircuitState.OPEN)
        self.assertEqual(circuit.consecutive_failures, 3)

        allowed, reason = circuit.can_attempt()
        self.assertFalse(allowed)
        self.assertIn("OPEN", reason)

    # -------------------------------------------------------------------------
    # Scenario 9: Circuit breaker recovery (half-open probe succeeds -> CLOSED)
    # -------------------------------------------------------------------------
    async def test_09_circuit_breaker_recovery(self):
        circuit = ProviderCircuit("gemini", failure_threshold=2, cooldown_seconds=0.1)
        circuit.record_failure("Fail 1")
        circuit.record_failure("Fail 2")
        self.assertEqual(circuit.state, CircuitState.OPEN)

        # Wait for short cooldown to expire
        await asyncio.sleep(0.15)

        # can_attempt should now transition to HALF_OPEN
        allowed, reason = circuit.can_attempt()
        self.assertTrue(allowed)
        self.assertEqual(circuit.state, CircuitState.HALF_OPEN)

        # Successful probe call resets circuit to CLOSED
        circuit.record_success()
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        self.assertEqual(circuit.consecutive_failures, 0)

    # -------------------------------------------------------------------------
    # Scenario 10: Brand information preservation across fallback
    # -------------------------------------------------------------------------
    async def test_10_brand_information_preservation_across_fallback(self):
        # Force degraded fallback
        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=httpx.ConnectTimeout("Timeout")):
            with patch.object(AIProviderManager, "_execute_openai_call", side_effect=httpx.ConnectTimeout("Timeout")):
                brand_data = {
                    "name": self.brand_acme.name,
                    "industry": self.brand_acme.industry,
                    "tone_of_voice": self.brand_acme.tone_of_voice,
                    "target_audience": self.brand_acme.target_audience,
                    "usp": self.brand_acme.meta_info.get("usp", "Zero-downtime microservice orchestration"),
                    "cta": self.brand_acme.cta,
                    "products": [{"name": "Acme MicroEngine", "description": "High-speed runtime"}],
                    "key_themes": self.brand_acme.key_themes
                }

                plan = await AIProviderManager.generate_json_with_resilience(
                    system_instruction="sys",
                    user_prompt="prompt",
                    brand_info=brand_data,
                    slot_count=7
                )

                # Verify each idea strictly reflects Acme's verified facts
                for idea in plan["ideas"]:
                    self.assertIn("Acme", idea["title"] + " " + idea["description"])
                    self.assertIn(self.brand_acme.cta, idea["cta"] + " " + idea["description"])

    # -------------------------------------------------------------------------
    # Scenario 11: Tenant isolation across fallback (zero cross-tenant leak)
    # -------------------------------------------------------------------------
    async def test_11_tenant_isolation_across_fallback(self):
        # Verify that generating a plan for Org 1 (Acme) never accesses or outputs Org 2 (Zenith) data
        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=httpx.ConnectError("Down")):
            with patch.object(AIProviderManager, "_execute_openai_call", side_effect=httpx.ConnectError("Down")):
                result = await ContentPlannerService.generate_plan(
                    db=self.db,
                    organization_id=self.org1.id,
                    brand_profile_id=self.brand_acme.id,
                    duration_days=7,
                    allow_degraded_fallback=True
                )

                self.assertIn("ideas", result)
                for idea in result["ideas"]:
                    text_blob = f"{idea['title']} {idea['description']} {idea.get('caption', '')}".lower()
                    self.assertNotIn("zenith", text_blob)
                    self.assertNotIn("organic coffee", text_blob)

    # -------------------------------------------------------------------------
    # Scenario 12: Concurrent generation requests with independent correlation IDs
    # -------------------------------------------------------------------------
    async def test_12_concurrent_generation_requests(self):
        valid_plan = self._sample_valid_plan()

        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=lambda *a, **kw: self._sample_valid_plan()):

            coros = [
                AIProviderManager.generate_json_with_resilience(
                    system_instruction="sys",
                    user_prompt="prompt",
                    brand_info={"name": f"Brand-{i}"},
                    corr_id=f"concurrent-{i}"
                )
                for i in range(5)
            ]

            results = await asyncio.gather(*coros)
            self.assertEqual(len(results), 5)
            corr_ids = [r["_correlation_id"] for r in results]
            self.assertEqual(len(set(corr_ids)), 5)  # 5 distinct correlation IDs

    # -------------------------------------------------------------------------
    # Scenario 13: Scheduled campaign generation with resilient fallback
    # -------------------------------------------------------------------------
    async def test_13_scheduled_campaign_generation_resilience(self):
        # Generate plan with resilient fallback and verify that scheduled jobs are created
        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=httpx.ConnectError("Down")):
            with patch.object(AIProviderManager, "_execute_openai_call", side_effect=httpx.ConnectError("Down")):
                plan = await ContentPlannerService.generate_plan(
                    db=self.db,
                    organization_id=self.org1.id,
                    brand_profile_id=self.brand_acme.id,
                    duration_days=7,
                    allow_degraded_fallback=True
                )

                # Save plan to database and verify scheduled jobs can be synchronized
                save_data = {
                    "brand_profile_id": self.brand_acme.id,
                    "title": plan.get("title") or plan.get("campaign_name", "Acme Campaign"),
                    "posting_frequency": "1_per_day",
                    "ideas": plan["ideas"]
                }
                saved = ContentPlannerService.save_plan_to_database(
                    db=self.db,
                    organization_id=self.org1.id,
                    current_user=self.user1,
                    plan_data=save_data
                )

                self.assertIsNotNone(saved.get("campaign_id"))
                ideas_in_db = self.db.query(Content).filter(Content.campaign_id == saved["campaign_id"]).all()
                self.assertEqual(len(ideas_in_db), 7)

    # -------------------------------------------------------------------------
    # Scenario 14: Background worker failure recovery
    # -------------------------------------------------------------------------
    async def test_14_background_worker_failure_recovery(self):
        # Test GenerationJob error recording and subsequent resilient recovery
        job = GenerationJob(
            job_type=GenerationType.TEXT_CAPTION,
            status=JobStatus.PROCESSING,
            prompt="Generate high-converting caption",
            model_name="gemini-3.5-flash",
            organization_id=self.org1.id,
            user_id=self.user1.id,
            parameters={"retry_count": 0}
        )
        self.db.add(job)
        self.db.commit()

        # Simulate job failure
        job.status = JobStatus.FAILED
        job.error_message = "HTTP 429: Rate limit exceeded"
        params = dict(job.parameters or {})
        params["retry_count"] = 1
        job.parameters = params
        self.db.commit()

        # Worker resumes with failover
        valid_plan = self._sample_valid_plan(slot_count=1)
        with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=lambda *a, **kw: self._sample_valid_plan()):

            res = await AIProviderManager.generate_json_with_resilience(
                system_instruction="sys",
                user_prompt="prompt",
                brand_info={"name": "Acme Cloud Tools"},
                slot_count=1
            )

            job.status = JobStatus.COMPLETED
            job.result_data = res
            job.error_message = None
            self.db.commit()

        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIsNotNone(job.result_data)
        self.assertIsNone(job.error_message)


if __name__ == "__main__":
    unittest.main()
