"""Read a saved brand, generate a real plan, save only to an in-memory database."""
import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import app.models
from app.db.base import Base
from app.db.session import SessionLocal
from app.models.brand_profile import BrandProfile
from app.models.organization import Organization
from app.models.user import User
from app.schemas.campaign import SavePlanRequest
from app.services.content_planner_service import ContentPlannerService as Planner
from app.services.gemini_service import GeminiService

class ProviderDiagnostics:
    def __init__(self, models):
        self.models = models

    def generate_content(self, **kwargs):
        try:
            result = self.models.generate_content(**kwargs)
            print('Gemini response text available:', bool(result.text))
            return result
        except Exception as exc:
            print('Gemini SDK failure:', type(exc).__name__, getattr(exc, 'code', ''), getattr(exc, 'status', ''))
            raise



async def main():
    # Only SELECT against application data. Persistence verification uses RAM.
    with SessionLocal() as source:
        brand = source.query(BrandProfile).order_by(BrandProfile.updated_at.desc()).first()
        if not brand:
            print('BLOCKED: No saved brand profile.')
            return
        values = {column.name: getattr(brand, column.name) for column in BrandProfile.__table__.columns if column.name not in ('id', 'organization_id', 'user_id', 'created_at', 'updated_at')}
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        org = Organization(name='Isolated live verification', slug='isolated-live')
        db.add(org)
        db.flush()
        user = User(email='live-check@example.test', hashed_password='unused', organization_id=org.id)
        brand = BrandProfile(**values, organization_id=org.id)
        db.add_all([brand, user])
        db.commit()
        client = GeminiService.get_client()
        if client:
            from types import SimpleNamespace
            GeminiService.get_client = classmethod(lambda cls: SimpleNamespace(models=ProviderDiagnostics(client.models)))
        plan = await Planner.generate_plan(db, org.id, brand.id, platforms=['linkedin'], duration_days=7)
        payload = SavePlanRequest(**plan)
        saved = Planner.save_plan_to_database(db, org.id, user, payload.model_dump())
        loaded = Planner.get_campaign_ideas(db, org.id, saved['campaign_id'])
        assert len(loaded['ideas']) == 7
        assert all(idea['description'] for idea in loaded['ideas'])
        print('PASS: saved brand -> live AI -> 7 ideas -> isolated persistence -> reload.')
    engine.dispose()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as exc:
        print('FAILED:', type(exc).__name__, getattr(exc, 'status_code', ''), getattr(exc, 'detail', 'No response details printed.'))
        sys.exit(1)
