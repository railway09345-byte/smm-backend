from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.approval import Approval
from app.models.content import Content
from app.models.brand_profile import BrandProfile
from app.models.user import User
from app.schemas.approval import ApprovalCreate, ApprovalReview
from app.utils.enums import ApprovalStatus, ContentStatus


class ApprovalService:
    @staticmethod
    def list_approvals(db: Session, organization_id: int, status_filter: Optional[ApprovalStatus] = None) -> List[Approval]:
        query = db.query(Approval).join(Content, Approval.content_id == Content.id).join(
            BrandProfile, Content.brand_profile_id == BrandProfile.id
        ).filter(BrandProfile.organization_id == organization_id)
        if status_filter:
            query = query.filter(Approval.status == status_filter)
        return query.order_by(Approval.created_at.desc()).all()

    @staticmethod
    def request_approval(db: Session, requester: User, payload: ApprovalCreate) -> Approval:
        content = db.query(Content).join(BrandProfile, Content.brand_profile_id == BrandProfile.id).filter(
            Content.id == payload.content_id,
            BrandProfile.organization_id == requester.organization_id
        ).first()
        if not content:
            foreign_content = db.query(Content).filter(Content.id == payload.content_id).first()
            if foreign_content:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Cross-tenant approval requests are strictly prohibited."
                )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Content not found.")

        content.status = ContentStatus.PENDING_APPROVAL

        approval = Approval(
            content_id=payload.content_id,
            requester_id=requester.id,
            comment=payload.comment,
            status=ApprovalStatus.PENDING
        )
        db.add(approval)
        db.commit()
        db.refresh(approval)
        return approval

    @staticmethod
    def review_approval(db: Session, approval_id: int, reviewer: User, payload: ApprovalReview) -> Approval:
        # 1. Enforce Role-Based Access Control: Only ADMIN, MANAGER, or SUPER_ADMIN may review
        reviewer_role = reviewer.role.value if hasattr(reviewer.role, "value") else str(reviewer.role).lower()
        if reviewer_role not in ["admin", "manager", "super_admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: User role '{reviewer_role}' lacks permission to review approvals. Only ADMIN or MANAGER roles can approve or reject content."
            )

        # 2. Strict Multi-Tenant Isolation Check
        approval = db.query(Approval).join(Content, Approval.content_id == Content.id).join(
            BrandProfile, Content.brand_profile_id == BrandProfile.id
        ).filter(
            Approval.id == approval_id,
            BrandProfile.organization_id == reviewer.organization_id
        ).first()

        if not approval:
            # Check if approval exists in a foreign organization to emit a strict 403 Forbidden
            foreign_approval = db.query(Approval).filter(Approval.id == approval_id).first()
            if foreign_approval:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Cross-tenant approval reviews are strictly prohibited."
                )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval request not found.")

        # 3. Segregation of Duties: Creator cannot approve their own content even if elevated
        if reviewer_role == "creator":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Creators are strictly prohibited from approving content."
            )

        approval.status = payload.status
        approval.comment = payload.comment or approval.comment
        approval.reviewer_id = reviewer.id
        approval.reviewed_at = datetime.utcnow()

        content = db.query(Content).filter(Content.id == approval.content_id).first()
        if content:
            if payload.status == ApprovalStatus.APPROVED:
                content.status = ContentStatus.APPROVED
                try:
                    from app.models.scheduled_post import ScheduledPost
                    linked_posts = db.query(ScheduledPost).filter(
                        ScheduledPost.content_id == content.id,
                        ScheduledPost.status.in_(["pending_approval", "awaiting", "draft"])
                    ).all()
                    for sp in linked_posts:
                        sp.status = "scheduled"
                except Exception:
                    pass
            elif payload.status == ApprovalStatus.REJECTED:
                content.status = ContentStatus.REJECTED
                try:
                    from app.models.scheduled_post import ScheduledPost
                    linked_posts = db.query(ScheduledPost).filter(
                        ScheduledPost.content_id == content.id,
                        ScheduledPost.status.in_(["pending_approval", "awaiting"])
                    ).all()
                    for sp in linked_posts:
                        sp.status = "rejected"
                except Exception:
                    pass

        db.commit()
        db.refresh(approval)
        return approval
