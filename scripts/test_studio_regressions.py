"""Isolated regression tests; never opens or deletes the application's database."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fastapi import HTTPException
import app.models
from app.db.base import Base
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.models.media_asset import MediaAsset
from app.models.quality_report import QualityReport
from app.schemas.campaign import SavePlanRequest
from app.services.content_planner_service import ContentPlannerService as Planner
from app.services.gemini_service import GeminiService
from app.services.ai_provider_manager import AIProviderManager
from app.services.image_generation_service import ImageGenerationService
from app.services.studio_brand_context import resolve_studio_brand, validate_studio_brand


class StudioRegressionTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        org = Organization(name='Test', slug='test')
        other = Organization(name='Other', slug='other')
        self.db.add_all([org, other])
        self.db.flush()
        self.user = User(email='studio@example.test', hashed_password='unused', organization_id=org.id)
        self.brand = BrandProfile(name='Test Software', industry='Software', brand_guidelines='Workflow automation services.', organization_id=org.id)
        self.foreign = BrandProfile(name='Other Brand', industry='Shoes', brand_guidelines='Running shoes.', organization_id=other.id)
        self.db.add_all([self.user, self.brand, self.foreign])
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_foreign_brand_is_rejected_without_fallback(self):
        with self.assertRaises(HTTPException):
            resolve_studio_brand(self.db, self.user.organization_id, self.foreign.id)

    def test_missing_business_facts_are_rejected(self):
        with self.assertRaises(HTTPException):
            validate_studio_brand(BrandProfile(name='Only a name'))

    def test_save_preserves_brand_frequency_focus_and_description(self):
        payload = SavePlanRequest(brand_profile_id=self.brand.id, title='Campaign', posting_frequency='2_per_day', custom_focus='Automation', ideas=[{'id': 'idea-d1-s1', 'title': 'Real idea', 'description': 'Complete caption', 'platforms': ['linkedin']}])
        result = Planner.save_plan_to_database(self.db, self.user.organization_id, self.user, payload.model_dump())
        campaign = self.db.get(Campaign, result['campaign_id'])
        self.assertEqual(campaign.brand_profile_id, self.brand.id)
        self.assertEqual(campaign.tags['posting_frequency'], '2_per_day')
        self.assertEqual(campaign.tags['custom_focus'], 'Automation')
        self.assertEqual(self.db.query(Content).one().caption, 'Complete caption')

    def test_save_rejects_foreign_brand_before_writing(self):
        with self.assertRaises(HTTPException):
            Planner.save_plan_to_database(self.db, self.user.organization_id, self.user, {'brand_profile_id': self.foreign.id, 'title': 'Forbidden'})
        self.assertEqual(self.db.query(Campaign).count(), 0)

    def test_provider_failure_never_returns_template_plan(self):
        with patch.object(AIProviderManager, 'generate_json_with_resilience', new=AsyncMock(return_value=None)), \
             patch.object(GeminiService, '_openai_json', new=AsyncMock(return_value=None)), patch.object(GeminiService, 'get_client', return_value=None):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(Planner.generate_plan(self.db, self.user.organization_id, self.brand.id))
        self.assertEqual(caught.exception.status_code, 503)

    def test_incomplete_ai_plan_is_rejected(self):
        with patch.object(AIProviderManager, 'generate_json_with_resilience', new=AsyncMock(return_value={'ideas': []})), \
             patch.object(GeminiService, '_openai_json', new=AsyncMock(return_value={'ideas': []})):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(Planner.generate_plan(self.db, self.user.organization_id, self.brand.id))
        self.assertEqual(caught.exception.status_code, 502)

    def test_failed_regeneration_preserves_saved_copy(self):
        saved = Planner.save_plan_to_database(self.db, self.user.organization_id, self.user, {
            'brand_profile_id': self.brand.id, 'title': 'Existing campaign',
            'ideas': [{'title': 'Existing idea', 'description': 'Preserve this caption'}],
        })
        idea_id = saved['ideas'][0]['id']
        with patch.object(AIProviderManager, 'generate_json_with_resilience', new=AsyncMock(return_value=None)), \
             patch.object(GeminiService, '_openai_json', new=AsyncMock(return_value=None)), patch.object(GeminiService, 'get_client', return_value=None):
            with self.assertRaises(HTTPException):
                asyncio.run(Planner.regenerate_single_idea(self.db, self.user.organization_id, idea_id))
        self.db.expire_all()
        self.assertEqual(self.db.get(Content, idea_id).caption, 'Preserve this caption')

    def test_scene_generation_does_not_substitute_stock_photography(self):
        with patch.object(ImageGenerationService, '_generate_cloudflare_image', new=AsyncMock(return_value=None)), patch.object(ImageGenerationService, '_generate_fal_image', new=AsyncMock(return_value=None)), patch.object(ImageGenerationService, '_generate_pollinations_image', new=AsyncMock(return_value=None)), patch.object(ImageGenerationService, '_fetch_commons_photoreal_image', new=AsyncMock()) as stock:
            result = asyncio.run(ImageGenerationService.generate_raw_image_bytes('Brand services', brand_profile=self.brand))
        self.assertIsNone(result)
        stock.assert_not_called()

    def test_generated_media_persists_and_foreign_media_is_rejected(self):
        asset = MediaAsset(file_name='unit.png', file_url='/unit.png', organization_id=self.user.organization_id)
        self.db.add(asset)
        self.db.flush()
        self.db.add(QualityReport(asset_type='image', media_asset_id=asset.id, organization_id=self.user.organization_id, brand_profile_id=self.brand.id))
        self.db.commit()
        saved = Planner.save_plan_to_database(self.db, self.user.organization_id, self.user, {
            'brand_profile_id': self.brand.id, 'title': 'Media campaign',
            'ideas': [{'title': 'Idea', 'description': 'Caption', 'media_asset_ids': [asset.id]}],
        })
        self.db.expire_all()
        loaded = Planner.get_campaign_ideas(self.db, self.user.organization_id, saved['campaign_id'])
        self.assertEqual(loaded['ideas'][0]['generated_media']['image']['media_asset_id'], asset.id)
        foreign = MediaAsset(file_name='foreign.png', file_url='/foreign.png', organization_id=self.foreign.organization_id)
        self.db.add(foreign)
        self.db.commit()
        with self.assertRaises(HTTPException):
            Planner.update_idea(self.db, self.user.organization_id, saved['ideas'][0]['id'], {'media_asset_ids': [foreign.id]})


if __name__ == '__main__':
    unittest.main(verbosity=2)
