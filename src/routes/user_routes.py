from typing import List, Set
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_
from src.config.db import get_db
from src.controllers.user_controller import UserController
from src.middleware.auth_middleware import get_current_user, require_admin
from src.models.user import User
from src.models.role import UserRole
from app.models.user_organization import UserOrganization
from src.validations.user_validation import UserResponseValidation, UserUpdateValidation

router = APIRouter(prefix="/users", tags=["Users"])


def _is_super_admin(user: User) -> bool:
    role_val = user.role.value if hasattr(user.role, 'value') else str(user.role).lower()
    return role_val in ("super_admin", "superadmin")


def _get_admin_accessible_org_ids(db: Session, admin_user: User) -> Set[int]:
    """
    Retrieve only organization IDs that this administrator is authorized to ADMINISTER.
    Ordinary memberships (CREATOR, MANAGER, VIEWER) do NOT grant administrative authority.
    """
    org_ids: Set[int] = set()

    # Primary organization: included only if admin_user has an administrative role globally
    user_role_str = admin_user.role.value if hasattr(admin_user.role, 'value') else str(admin_user.role).lower()
    if user_role_str in ("admin", "super_admin", "superadmin") and admin_user.organization_id:
        org_ids.add(admin_user.organization_id)

    # In UserOrganization, include only organizations where the user has an explicit ADMIN or SUPER_ADMIN role
    admin_roles = [UserRole.ADMIN, UserRole.SUPER_ADMIN, "admin", "super_admin"]
    uos = db.query(UserOrganization.organization_id).filter(
        UserOrganization.user_id == admin_user.id,
        UserOrganization.role.in_(admin_roles)
    ).all()
    for uo in uos:
        if uo[0]:
            org_ids.add(uo[0])
    return org_ids


def _verify_user_in_org(db: Session, admin_user: User, target_user: User) -> bool:
    """Verify if target_user belongs to an organization managed by admin_user."""
    if _is_super_admin(admin_user):
        return True
    admin_org_ids = _get_admin_accessible_org_ids(db, admin_user)
    if not admin_org_ids:
        return False
    if target_user.organization_id and target_user.organization_id in admin_org_ids:
        return True
    has_uo = db.query(UserOrganization).filter(
        UserOrganization.user_id == target_user.id,
        UserOrganization.organization_id.in_(admin_org_ids)
    ).first()
    return has_uo is not None


@router.get("/me", response_model=UserResponseValidation)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Retrieve current authenticated user."""
    return UserController.get_profile(current_user)


@router.get("/", response_model=List[UserResponseValidation])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Admin endpoint to list all users within the admin's organization scope."""
    if _is_super_admin(current_user):
        return UserController.list_users(db)

    admin_org_ids = _get_admin_accessible_org_ids(db, current_user)
    if not admin_org_ids:
        return []

    users = db.query(User).filter(
        or_(
            User.organization_id.in_(admin_org_ids),
            User.id.in_(
                db.query(UserOrganization.user_id).filter(
                    UserOrganization.organization_id.in_(admin_org_ids)
                )
            )
        ),
        User.role != UserRole.SUPER_ADMIN
    ).order_by(User.created_at.desc()).all()
    return users


@router.get("/{user_id}", response_model=UserResponseValidation)
def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve user by ID with strict organization boundary checks."""
    from src.services.user_service import UserService
    target_user = UserService.get_by_id(db, user_id)

    if current_user.id != target_user.id:
        if not _is_super_admin(current_user):
            user_role_str = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
            if user_role_str not in ("admin", "super_admin"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions to view other users."
                )
            if _is_super_admin(target_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Enterprise Admins cannot access SuperAdmin accounts."
                )
            if not _verify_user_in_org(db, current_user, target_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Target user does not belong to your organization."
                )

    return UserController.get_user_by_id(db, user_id)


@router.put("/{user_id}", response_model=UserResponseValidation)
def update_user(
    user_id: int,
    data: UserUpdateValidation,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update user information with RBAC and organization security checks."""
    from src.services.user_service import UserService
    target_user = UserService.get_by_id(db, user_id)

    user_role_str = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
    is_current_admin = user_role_str in ("admin", "super_admin", "superadmin")
    is_current_super = _is_super_admin(current_user)

    # 1. Protect SuperAdmin accounts from modification by non-superadmins
    if _is_super_admin(target_user) and not is_current_super:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Enterprise Admins cannot modify SuperAdmin accounts."
        )

    # 2. Role change permissions & self-modification restrictions
    if data.role is not None:
        if not is_current_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to modify user roles."
            )
        if current_user.id == target_user.id and not is_current_super:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot change your own role."
            )
        role_str = data.role.value if hasattr(data.role, 'value') else str(data.role).lower()
        if role_str in ("super_admin", "superadmin") and not is_current_super:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Enterprise Admins cannot assign or promote to the SuperAdmin role."
            )

    # 3. Account status change permissions
    if data.is_active is not None:
        if not is_current_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can modify account status."
            )
        if current_user.id == target_user.id and not data.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot deactivate your own account."
            )

    # 4. Organization boundary enforcement when modifying other users
    if current_user.id != target_user.id:
        if not is_current_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to modify other users."
            )
        if not is_current_super:
            if not _verify_user_in_org(db, current_user, target_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Target user does not belong to your organization."
                )

    # 5. Prohibit moving users to organizations outside admin's boundary
    if data.organization_id is not None and data.organization_id != target_user.organization_id:
        if not is_current_super:
            if not is_current_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only administrators can modify user organization assignment."
                )
            admin_org_ids = _get_admin_accessible_org_ids(db, current_user)
            if data.organization_id not in admin_org_ids:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot assign user to an organization outside your management."
                )

    return UserController.update_user(db, user_id, data)


@router.delete("/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """
    Delete a global user account.
    Security Enforcement: Only SuperAdmins may permanently delete global user accounts.
    Enterprise Admins must use DELETE /api/v1/admin/users/{user_id}/unassign-business/{business_id}
    to remove members from their organization.
    """
    if not _is_super_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Only SuperAdmins can permanently delete global user accounts. Use unassign-business to remove members from your organization."
        )

    from src.services.user_service import UserService
    target_user = UserService.get_by_id(db, user_id)

    if current_user.id == target_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account."
        )

    return UserController.delete_user(db, user_id)


__all__ = ["router"]
