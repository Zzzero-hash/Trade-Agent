"""
KYC/AML Service with Jumio and Onfido integration for identity verification.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import httpx
from pydantic import BaseModel, Field

from src.models.compliance import KYCData, KYCStatus, AMLStatus, DocumentType
from src.config.settings import get_settings
from src.services.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class KYCProvider(str, Enum):
    JUMIO = "jumio"
    ONFIDO = "onfido"


class KYCRequest(BaseModel):
    customer_id: str
    document_type: DocumentType
    document_front: bytes
    document_back: Optional[bytes] = None
    selfie: Optional[bytes] = None
    provider: KYCProvider = KYCProvider.JUMIO


class KYCResult(BaseModel):
    verification_id: str
    status: KYCStatus
    confidence_score: float
    extracted_data: Dict[str, Any]
    verification_timestamp: datetime
    provider: KYCProvider


class JumioClient:
    """Jumio API client for identity verification."""
    
    def __init__(self, api_token: str, api_secret: str, base_url: str):
        self.api_token = api_token
        self.api_secret = api_secret
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            auth=(api_token, api_secret),
            timeout=30.0
        )
    
    async def create_verification(self, request: KYCRequest) -> str:
        """Create a new verification session with Jumio."""
        payload = {
            "customerInternalReference": request.customer_id,
            "userReference": f"user_{request.customer_id}",
            "workflowId": "100",  # Standard ID verification workflow
            "presets": [
                {
                    "index": 1,
                    "country": "USA",
                    "type": "ID_CARD"
                }
            ]
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v4/initiate",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["transactionReference"]
    
    async def get_verification_result(self, transaction_ref: str) -> KYCResult:
        """Get verification result from Jumio."""
        response = await self.client.get(
            f"{self.base_url}/api/v4/accounts/{transaction_ref}"
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Map Jumio status to our KYC status
        jumio_status = data.get("status", "").upper()
        if jumio_status == "APPROVED_VERIFIED":
            status = KYCStatus.APPROVED
        elif jumio_status == "DENIED_FRAUD":
            status = KYCStatus.REJECTED
        else:
            status = KYCStatus.PENDING
        
        return KYCResult(
            verification_id=transaction_ref,
            status=status,
            confidence_score=data.get("verificationScore", 0.0) / 100.0,
            extracted_data=data.get("document", {}),
            verification_timestamp=datetime.utcnow(),
            provider=KYCProvider.JUMIO
        )


class OnfidoClient:
    """Onfido API client for identity verification."""
    
    def __init__(self, api_token: str, base_url: str):
        self.api_token = api_token
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Token token={api_token}"},
            timeout=30.0
        )
    
    async def create_applicant(self, customer_id: str) -> str:
        """Create an applicant in Onfido."""
        payload = {
            "first_name": "Customer",
            "last_name": customer_id,
            "email": f"customer_{customer_id}@example.com"
        }
        
        response = await self.client.post(
            f"{self.base_url}/v3.6/applicants",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["id"]
    
    async def upload_document(self, applicant_id: str, document: bytes, doc_type: str) -> str:
        """Upload document to Onfido."""
        files = {
            "file": ("document.jpg", document, "image/jpeg"),
            "type": (None, doc_type)
        }
        
        response = await self.client.post(
            f"{self.base_url}/v3.6/documents",
            files=files,
            data={"applicant_id": applicant_id}
        )
        response.raise_for_status()
        
        result = response.json()
        return result["id"]
    
    async def create_check(self, applicant_id: str) -> str:
        """Create identity check in Onfido."""
        payload = {
            "applicant_id": applicant_id,
            "report_names": ["identity_enhanced"]
        }
        
        response = await self.client.post(
            f"{self.base_url}/v3.6/checks",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["id"]
    
    async def get_check_result(self, check_id: str) -> KYCResult:
        """Get check result from Onfido."""
        response = await self.client.get(
            f"{self.base_url}/v3.6/checks/{check_id}"
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Map Onfido status to our KYC status
        onfido_status = data.get("status", "").lower()
        if onfido_status == "complete" and data.get("result") == "clear":
            status = KYCStatus.APPROVED
        elif data.get("result") == "consider":
            status = KYCStatus.MANUAL_REVIEW
        else:
            status = KYCStatus.PENDING
        
        return KYCResult(
            verification_id=check_id,
            status=status,
            confidence_score=0.95 if status == KYCStatus.APPROVED else 0.5,
            extracted_data=data,
            verification_timestamp=datetime.utcnow(),
            provider=KYCProvider.ONFIDO
        )


class KYCService:
    """Main KYC service orchestrating multiple providers."""
    
    def __init__(self, encryption_service: EncryptionService):
        self.settings = get_settings()
        self.encryption_service = encryption_service
        
        # Initialize providers
        self.jumio_client = JumioClient(
            api_token=self.settings.jumio_api_token,
            api_secret=self.settings.jumio_api_secret,
            base_url=self.settings.jumio_base_url
        )
        
        self.onfido_client = OnfidoClient(
            api_token=self.settings.onfido_api_token,
            base_url=self.settings.onfido_base_url
        )
    
    async def initiate_verification(self, request: KYCRequest) -> str:
        """Initiate KYC verification with specified provider."""
        try:
            if request.provider == KYCProvider.JUMIO:
                verification_id = await self.jumio_client.create_verification(request)
            else:
                # Onfido workflow
                applicant_id = await self.onfido_client.create_applicant(request.customer_id)
                doc_id = await self.onfido_client.upload_document(
                    applicant_id, request.document_front, "passport"
                )
                verification_id = await self.onfido_client.create_check(applicant_id)
            
            logger.info(f"KYC verification initiated: {verification_id} for customer {request.customer_id}")
            return verification_id
            
        except Exception as e:
            logger.error(f"KYC verification failed for customer {request.customer_id}: {str(e)}")
            raise
    
    async def get_verification_status(self, verification_id: str, provider: KYCProvider) -> KYCResult:
        """Get verification status from provider."""
        try:
            if provider == KYCProvider.JUMIO:
                result = await self.jumio_client.get_verification_result(verification_id)
            else:
                result = await self.onfido_client.get_check_result(verification_id)
            
            # Encrypt sensitive data before storage
            if result.extracted_data:
                result.extracted_data = await self.encryption_service.encrypt_data(
                    result.extracted_data
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get verification status for {verification_id}: {str(e)}")
            raise
    
    async def process_webhook(self, provider: KYCProvider, payload: Dict[str, Any]) -> Optional[KYCResult]:
        """Process webhook notifications from KYC providers."""
        try:
            if provider == KYCProvider.JUMIO:
                return await self._process_jumio_webhook(payload)
            else:
                return await self._process_onfido_webhook(payload)
        except Exception as e:
            logger.error(f"Webhook processing failed for {provider}: {str(e)}")
            return None
    
    async def _process_jumio_webhook(self, payload: Dict[str, Any]) -> Optional[KYCResult]:
        """Process Jumio webhook payload."""
        transaction_ref = payload.get("transactionReference")
        if not transaction_ref:
            return None
        
        return await self.jumio_client.get_verification_result(transaction_ref)
    
    async def _process_onfido_webhook(self, payload: Dict[str, Any]) -> Optional[KYCResult]:
        """Process Onfido webhook payload."""
        check_id = payload.get("object", {}).get("id")
        if not check_id:
            return None
        
        return await self.onfido_client.get_check_result(check_id)