"""KYC/AML service with Jumio and Onfido integration."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel

from src.models.security import (
    KYCData, KYCDocument, KYCStatus, DocumentType, 
    VerificationProvider, AMLStatus, OFACScreeningResult, PEPScreeningResult
)
from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class JumioConfig(BaseModel):
    """Jumio API configuration."""
    api_token: str
    api_secret: str
    base_url: str = "https://netverify.com/api/netverify/v2"
    callback_url: Optional[str] = None


class OnfidoConfig(BaseModel):
    """Onfido API configuration."""
    api_token: str
    base_url: str = "https://api.onfido.com/v3.6"
    webhook_token: Optional[str] = None


class KYCService:
    """KYC/AML service for identity verification and compliance screening."""
    
    def __init__(self, jumio_config: Optional[JumioConfig] = None, 
                 onfido_config: Optional[OnfidoConfig] = None):
        """Initialize KYC service with provider configurations."""
        self.jumio_config = jumio_config
        self.onfido_config = onfido_config
        self.settings = get_settings()
        
        # Initialize HTTP clients
        self.jumio_client = None
        self.onfido_client = None
        
        if jumio_config:
            self.jumio_client = httpx.AsyncClient(
                base_url=jumio_config.base_url,
                auth=(jumio_config.api_token, jumio_config.api_secret),
                timeout=30.0
            )
            
        if onfido_config:
            self.onfido_client = httpx.AsyncClient(
                base_url=onfido_config.base_url,
                headers={"Authorization": f"Token token={onfido_config.api_token}"},
                timeout=30.0
            )
    
    async def initiate_kyc_verification(self, customer_id: UUID, 
                                      provider: VerificationProvider,
                                      customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate KYC verification with specified provider."""
        try:
            if provider == VerificationProvider.JUMIO:
                return await self._initiate_jumio_verification(customer_id, customer_data)
            elif provider == VerificationProvider.ONFIDO:
                return await self._initiate_onfido_verification(customer_id, customer_data)
            else:
                raise ValueError(f"Unsupported verification provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to initiate KYC verification: {e}")
            raise
    
    async def _initiate_jumio_verification(self, customer_id: UUID, 
                                         customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate verification with Jumio."""
        if not self.jumio_client:
            raise ValueError("Jumio client not configured")
        
        payload = {
            "customerInternalReference": str(customer_id),
            "userReference": customer_data.get("email"),
            "callbackUrl": self.jumio_config.callback_url,
            "successUrl": f"{self.settings.frontend_url}/kyc/success",
            "errorUrl": f"{self.settings.frontend_url}/kyc/error",
            "enabledFields": "idNumber,idFirstName,idLastName,idDob,idExpiry,idUsState,idPersonalNumber,idFaceMatch",
            "authorizationTokenLifetime": 5184000  # 60 days
        }
        
        response = await self.jumio_client.post("/initiateNetverify", json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Store verification session
        await self._store_verification_session(
            customer_id=customer_id,
            provider=VerificationProvider.JUMIO,
            session_data=result
        )
        
        return {
            "verification_url": result.get("redirectUrl"),
            "session_id": result.get("jumioIdScanReference"),
            "provider": VerificationProvider.JUMIO
        }
    
    async def _initiate_onfido_verification(self, customer_id: UUID,
                                          customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate verification with Onfido."""
        if not self.onfido_client:
            raise ValueError("Onfido client not configured")
        
        # Create applicant
        applicant_payload = {
            "first_name": customer_data.get("first_name"),
            "last_name": customer_data.get("last_name"),
            "email": customer_data.get("email"),
            "dob": customer_data.get("date_of_birth"),
            "address": customer_data.get("address", {})
        }
        
        response = await self.onfido_client.post("/applicants", json=applicant_payload)
        response.raise_for_status()
        applicant = response.json()
        
        # Create SDK token
        token_payload = {
            "applicant_id": applicant["id"],
            "application_id": str(customer_id)
        }
        
        response = await self.onfido_client.post("/sdk_token", json=token_payload)
        response.raise_for_status()
        token_result = response.json()
        
        # Store verification session
        await self._store_verification_session(
            customer_id=customer_id,
            provider=VerificationProvider.ONFIDO,
            session_data={
                "applicant_id": applicant["id"],
                "sdk_token": token_result["token"]
            }
        )
        
        return {
            "sdk_token": token_result["token"],
            "applicant_id": applicant["id"],
            "provider": VerificationProvider.ONFIDO
        }
    
    async def process_verification_callback(self, provider: VerificationProvider,
                                          callback_data: Dict[str, Any]) -> KYCData:
        """Process verification callback from provider."""
        try:
            if provider == VerificationProvider.JUMIO:
                return await self._process_jumio_callback(callback_data)
            elif provider == VerificationProvider.ONFIDO:
                return await self._process_onfido_callback(callback_data)
            else:
                raise ValueError(f"Unsupported verification provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to process verification callback: {e}")
            raise
    
    async def _process_jumio_callback(self, callback_data: Dict[str, Any]) -> KYCData:
        """Process Jumio verification callback."""
        scan_reference = callback_data.get("jumioIdScanReference")
        verification_status = callback_data.get("verificationStatus")
        
        # Retrieve detailed scan data
        response = await self.jumio_client.get(f"/scans/{scan_reference}")
        response.raise_for_status()
        scan_data = response.json()
        
        # Map Jumio status to our KYC status
        status_mapping = {
            "APPROVED_VERIFIED": KYCStatus.VERIFIED,
            "DENIED_FRAUD": KYCStatus.REJECTED,
            "DENIED_UNSUPPORTED_ID_TYPE": KYCStatus.REJECTED,
            "ERROR_NOT_READABLE_ID": KYCStatus.REJECTED,
            "NO_ID_UPLOADED": KYCStatus.REJECTED
        }
        
        kyc_status = status_mapping.get(verification_status, KYCStatus.PENDING)
        
        # Extract customer ID from internal reference
        customer_id = UUID(callback_data.get("customerInternalReference"))
        
        # Create KYC data
        kyc_data = KYCData(
            customer_id=customer_id,
            kyc_status=kyc_status,
            identity_verified=(kyc_status == KYCStatus.VERIFIED),
            address_verified=scan_data.get("addressVerification") == "MATCH",
            verification_provider=VerificationProvider.JUMIO,
            last_updated=datetime.utcnow()
        )
        
        # Store KYC data
        await self._store_kyc_data(kyc_data)
        
        # Trigger AML screening if verified
        if kyc_status == KYCStatus.VERIFIED:
            await self.perform_aml_screening(customer_id, scan_data)
        
        return kyc_data
    
    async def _process_onfido_callback(self, callback_data: Dict[str, Any]) -> KYCData:
        """Process Onfido verification callback."""
        check_id = callback_data.get("object", {}).get("id")
        
        # Retrieve check details
        response = await self.onfido_client.get(f"/checks/{check_id}")
        response.raise_for_status()
        check_data = response.json()
        
        # Map Onfido status to our KYC status
        result = check_data.get("result")
        status_mapping = {
            "clear": KYCStatus.VERIFIED,
            "consider": KYCStatus.UNDER_REVIEW,
            "unidentified": KYCStatus.REJECTED
        }
        
        kyc_status = status_mapping.get(result, KYCStatus.PENDING)
        
        # Extract customer ID from application ID
        customer_id = UUID(check_data.get("applicant_id"))
        
        # Create KYC data
        kyc_data = KYCData(
            customer_id=customer_id,
            kyc_status=kyc_status,
            identity_verified=(kyc_status == KYCStatus.VERIFIED),
            verification_provider=VerificationProvider.ONFIDO,
            last_updated=datetime.utcnow()
        )
        
        # Store KYC data
        await self._store_kyc_data(kyc_data)
        
        # Trigger AML screening if verified
        if kyc_status == KYCStatus.VERIFIED:
            await self.perform_aml_screening(customer_id, check_data)
        
        return kyc_data
    
    async def perform_aml_screening(self, customer_id: UUID, 
                                  identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AML screening including OFAC and PEP checks."""
        try:
            # Extract name and other identifying information
            first_name = identity_data.get("firstName") or identity_data.get("first_name")
            last_name = identity_data.get("lastName") or identity_data.get("last_name")
            date_of_birth = identity_data.get("dob") or identity_data.get("date_of_birth")
            
            # Perform OFAC screening
            ofac_result = await self._screen_ofac(
                customer_id=customer_id,
                first_name=first_name,
                last_name=last_name,
                date_of_birth=date_of_birth
            )
            
            # Perform PEP screening
            pep_result = await self._screen_pep(
                customer_id=customer_id,
                first_name=first_name,
                last_name=last_name,
                date_of_birth=date_of_birth
            )
            
            # Determine overall AML status
            aml_status = AMLStatus.CLEAR
            if ofac_result.is_match or pep_result.is_pep:
                aml_status = AMLStatus.FLAGGED
            
            # Update KYC data with AML results
            await self._update_aml_status(customer_id, aml_status)
            
            return {
                "aml_status": aml_status,
                "ofac_result": ofac_result,
                "pep_result": pep_result
            }
            
        except Exception as e:
            logger.error(f"AML screening failed for customer {customer_id}: {e}")
            await self._update_aml_status(customer_id, AMLStatus.UNDER_REVIEW)
            raise
    
    async def _screen_ofac(self, customer_id: UUID, first_name: str, 
                          last_name: str, date_of_birth: Optional[str]) -> OFACScreeningResult:
        """Screen against OFAC sanctions list."""
        # This would integrate with a real OFAC screening service
        # For now, implementing a mock screening
        
        screening_id = uuid4()
        
        # Mock screening logic - in production, use actual OFAC API
        full_name = f"{first_name} {last_name}".lower()
        
        # Simple mock check against known test names
        sanctioned_names = ["john doe", "jane smith", "test user"]
        is_match = any(name in full_name for name in sanctioned_names)
        
        result = OFACScreeningResult(
            customer_id=customer_id,
            screening_id=screening_id,
            is_match=is_match,
            match_score=0.95 if is_match else 0.0,
            matched_entries=[{"name": full_name, "list": "SDN"}] if is_match else [],
            screened_at=datetime.utcnow(),
            list_version="20240101"
        )
        
        # Store screening result
        await self._store_ofac_result(result)
        
        return result
    
    async def _screen_pep(self, customer_id: UUID, first_name: str,
                         last_name: str, date_of_birth: Optional[str]) -> PEPScreeningResult:
        """Screen against PEP (Politically Exposed Person) list."""
        # This would integrate with a real PEP screening service
        
        screening_id = uuid4()
        
        # Mock screening logic
        full_name = f"{first_name} {last_name}".lower()
        
        # Simple mock check
        pep_names = ["political figure", "government official"]
        is_pep = any(name in full_name for name in pep_names)
        
        result = PEPScreeningResult(
            customer_id=customer_id,
            screening_id=screening_id,
            is_pep=is_pep,
            pep_category="government" if is_pep else None,
            risk_level="high" if is_pep else None,
            matched_entries=[{"name": full_name, "category": "government"}] if is_pep else [],
            screened_at=datetime.utcnow(),
            list_version="20240101"
        )
        
        # Store screening result
        await self._store_pep_result(result)
        
        return result
    
    async def get_kyc_status(self, customer_id: UUID) -> Optional[KYCData]:
        """Get KYC status for a customer."""
        # This would query the database
        # For now, returning None - implement with actual database integration
        return None
    
    async def update_kyc_document(self, customer_id: UUID, document: KYCDocument) -> None:
        """Update KYC document information."""
        # Store document metadata
        logger.info(f"Storing KYC document for customer {customer_id}: {document.document_type}")
        # Implement database storage
    
    async def _store_verification_session(self, customer_id: UUID, 
                                        provider: VerificationProvider,
                                        session_data: Dict[str, Any]) -> None:
        """Store verification session data."""
        logger.info(f"Storing verification session for customer {customer_id} with {provider}")
        # Implement database storage
    
    async def _store_kyc_data(self, kyc_data: KYCData) -> None:
        """Store KYC data in database."""
        logger.info(f"Storing KYC data for customer {kyc_data.customer_id}")
        # Implement database storage
    
    async def _store_ofac_result(self, result: OFACScreeningResult) -> None:
        """Store OFAC screening result."""
        logger.info(f"Storing OFAC result for customer {result.customer_id}")
        # Implement database storage
    
    async def _store_pep_result(self, result: PEPScreeningResult) -> None:
        """Store PEP screening result."""
        logger.info(f"Storing PEP result for customer {result.customer_id}")
        # Implement database storage
    
    async def _update_aml_status(self, customer_id: UUID, status: AMLStatus) -> None:
        """Update AML status for customer."""
        logger.info(f"Updating AML status for customer {customer_id}: {status}")
        # Implement database update
    
    async def close(self) -> None:
        """Close HTTP clients."""
        if self.jumio_client:
            await self.jumio_client.aclose()
        if self.onfido_client:
            await self.onfido_client.aclose()