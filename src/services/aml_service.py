"""
AML Service with OFAC and PEP list screening and automated updates.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import httpx
import xml.etree.ElementTree as ET
from pydantic import BaseModel

from src.models.compliance import AMLStatus, AMLScreeningResult
from src.config.settings import get_settings
from src.repositories.compliance_repository import ComplianceRepository

logger = logging.getLogger(__name__)


class ScreeningListType(str, Enum):
    OFAC_SDN = "ofac_sdn"
    OFAC_CONSOLIDATED = "ofac_consolidated"
    PEP_LIST = "pep_list"
    EU_SANCTIONS = "eu_sanctions"


class ScreeningMatch(BaseModel):
    list_type: ScreeningListType
    match_score: float
    matched_name: str
    matched_entity: Dict
    confidence_level: str


class AMLScreeningRequest(BaseModel):
    customer_id: str
    full_name: str
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    address: Optional[str] = None


class OFACClient:
    """Client for OFAC sanctions list screening."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.sdn_url = "https://www.treasury.gov/ofac/downloads/sdn.xml"
        self.consolidated_url = "https://www.treasury.gov/ofac/downloads/consolidated/consolidated.xml"
    
    async def download_sdn_list(self) -> List[Dict]:
        """Download and parse OFAC SDN list."""
        try:
            response = await self.client.get(self.sdn_url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            entities = []
            
            for sdn_entry in root.findall(".//sdnEntry"):
                entity = {
                    "uid": sdn_entry.get("uid"),
                    "first_name": "",
                    "last_name": "",
                    "full_name": "",
                    "sdn_type": sdn_entry.get("sdnType"),
                    "programs": []
                }
                
                # Extract names
                for name in sdn_entry.findall(".//firstName"):
                    entity["first_name"] = name.text or ""
                
                for name in sdn_entry.findall(".//lastName"):
                    entity["last_name"] = name.text or ""
                
                # Build full name
                entity["full_name"] = f"{entity['first_name']} {entity['last_name']}".strip()
                
                # Extract programs
                for program in sdn_entry.findall(".//program"):
                    entity["programs"].append(program.text)
                
                entities.append(entity)
            
            logger.info(f"Downloaded {len(entities)} OFAC SDN entries")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to download OFAC SDN list: {str(e)}")
            raise
    
    async def download_consolidated_list(self) -> List[Dict]:
        """Download and parse OFAC consolidated list."""
        try:
            response = await self.client.get(self.consolidated_url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            entities = []
            
            for entry in root.findall(".//sdnEntry"):
                entity = {
                    "uid": entry.get("uid"),
                    "names": [],
                    "addresses": [],
                    "programs": [],
                    "remarks": ""
                }
                
                # Extract all name variations
                for name in entry.findall(".//aka"):
                    entity["names"].append(name.text)
                
                # Extract addresses
                for addr in entry.findall(".//address"):
                    address_parts = []
                    for part in addr:
                        if part.text:
                            address_parts.append(part.text)
                    entity["addresses"].append(" ".join(address_parts))
                
                # Extract programs
                for program in entry.findall(".//program"):
                    entity["programs"].append(program.text)
                
                entities.append(entity)
            
            logger.info(f"Downloaded {len(entities)} OFAC consolidated entries")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to download OFAC consolidated list: {str(e)}")
            raise


class PEPClient:
    """Client for PEP (Politically Exposed Persons) list screening."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.settings.pep_api_key}"},
            timeout=30.0
        )
        self.base_url = self.settings.pep_api_base_url
    
    async def download_pep_list(self) -> List[Dict]:
        """Download PEP list from commercial provider."""
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/pep-list")
            response.raise_for_status()
            
            data = response.json()
            entities = data.get("entities", [])
            
            logger.info(f"Downloaded {len(entities)} PEP entries")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to download PEP list: {str(e)}")
            raise
    
    async def search_pep(self, name: str, country: Optional[str] = None) -> List[Dict]:
        """Search PEP database for specific person."""
        try:
            params = {"name": name}
            if country:
                params["country"] = country
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/search",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("matches", [])
            
        except Exception as e:
            logger.error(f"PEP search failed for {name}: {str(e)}")
            return []


class NameMatcher:
    """Fuzzy name matching for sanctions screening."""
    
    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        """Calculate similarity score between two names."""
        # Simple Levenshtein distance implementation
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        if name1 == name2:
            return 1.0
        
        # Calculate Levenshtein distance
        len1, len2 = len(name1), len(name2)
        if len1 == 0:
            return 0.0
        if len2 == 0:
            return 0.0
        
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if name1[i-1] == name2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    @staticmethod
    def is_potential_match(name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two names are potential matches."""
        similarity = NameMatcher.calculate_similarity(name1, name2)
        return similarity >= threshold


class AMLService:
    """Main AML service for sanctions and PEP screening."""
    
    def __init__(self, compliance_repo: ComplianceRepository):
        self.compliance_repo = compliance_repo
        self.ofac_client = OFACClient()
        self.pep_client = PEPClient()
        self.name_matcher = NameMatcher()
        
        # In-memory cache for screening lists
        self.screening_lists: Dict[ScreeningListType, List[Dict]] = {}
        self.last_update: Dict[ScreeningListType, datetime] = {}
    
    async def update_screening_lists(self) -> None:
        """Update all screening lists from external sources."""
        try:
            # Update OFAC lists
            sdn_list = await self.ofac_client.download_sdn_list()
            self.screening_lists[ScreeningListType.OFAC_SDN] = sdn_list
            self.last_update[ScreeningListType.OFAC_SDN] = datetime.utcnow()
            
            consolidated_list = await self.ofac_client.download_consolidated_list()
            self.screening_lists[ScreeningListType.OFAC_CONSOLIDATED] = consolidated_list
            self.last_update[ScreeningListType.OFAC_CONSOLIDATED] = datetime.utcnow()
            
            # Update PEP list
            pep_list = await self.pep_client.download_pep_list()
            self.screening_lists[ScreeningListType.PEP_LIST] = pep_list
            self.last_update[ScreeningListType.PEP_LIST] = datetime.utcnow()
            
            # Store in database for persistence
            await self.compliance_repo.store_screening_lists(self.screening_lists)
            
            logger.info("Successfully updated all screening lists")
            
        except Exception as e:
            logger.error(f"Failed to update screening lists: {str(e)}")
            raise
    
    async def screen_customer(self, request: AMLScreeningRequest) -> AMLScreeningResult:
        """Screen customer against all sanctions and PEP lists."""
        try:
            matches = []
            
            # Ensure lists are up to date
            await self._ensure_lists_updated()
            
            # Screen against OFAC SDN list
            ofac_matches = await self._screen_against_ofac_sdn(request.full_name)
            matches.extend(ofac_matches)
            
            # Screen against OFAC consolidated list
            consolidated_matches = await self._screen_against_ofac_consolidated(request.full_name)
            matches.extend(consolidated_matches)
            
            # Screen against PEP list
            pep_matches = await self._screen_against_pep_list(request.full_name)
            matches.extend(pep_matches)
            
            # Determine overall status
            if not matches:
                status = AMLStatus.CLEAR
            elif any(match.match_score > 0.9 for match in matches):
                status = AMLStatus.BLOCKED
            else:
                status = AMLStatus.MANUAL_REVIEW
            
            result = AMLScreeningResult(
                customer_id=request.customer_id,
                status=status,
                matches=matches,
                screening_timestamp=datetime.utcnow(),
                lists_version=self._get_lists_version()
            )
            
            # Store result
            await self.compliance_repo.store_aml_result(result)
            
            logger.info(f"AML screening completed for customer {request.customer_id}: {status}")
            return result
            
        except Exception as e:
            logger.error(f"AML screening failed for customer {request.customer_id}: {str(e)}")
            raise
    
    async def _screen_against_ofac_sdn(self, name: str) -> List[ScreeningMatch]:
        """Screen name against OFAC SDN list."""
        matches = []
        sdn_list = self.screening_lists.get(ScreeningListType.OFAC_SDN, [])
        
        for entity in sdn_list:
            entity_name = entity.get("full_name", "")
            if not entity_name:
                continue
            
            similarity = self.name_matcher.calculate_similarity(name, entity_name)
            if similarity >= 0.7:  # Threshold for potential match
                match = ScreeningMatch(
                    list_type=ScreeningListType.OFAC_SDN,
                    match_score=similarity,
                    matched_name=entity_name,
                    matched_entity=entity,
                    confidence_level="high" if similarity > 0.9 else "medium"
                )
                matches.append(match)
        
        return matches
    
    async def _screen_against_ofac_consolidated(self, name: str) -> List[ScreeningMatch]:
        """Screen name against OFAC consolidated list."""
        matches = []
        consolidated_list = self.screening_lists.get(ScreeningListType.OFAC_CONSOLIDATED, [])
        
        for entity in consolidated_list:
            for entity_name in entity.get("names", []):
                if not entity_name:
                    continue
                
                similarity = self.name_matcher.calculate_similarity(name, entity_name)
                if similarity >= 0.7:
                    match = ScreeningMatch(
                        list_type=ScreeningListType.OFAC_CONSOLIDATED,
                        match_score=similarity,
                        matched_name=entity_name,
                        matched_entity=entity,
                        confidence_level="high" if similarity > 0.9 else "medium"
                    )
                    matches.append(match)
        
        return matches
    
    async def _screen_against_pep_list(self, name: str) -> List[ScreeningMatch]:
        """Screen name against PEP list."""
        matches = []
        pep_list = self.screening_lists.get(ScreeningListType.PEP_LIST, [])
        
        for entity in pep_list:
            entity_name = entity.get("name", "")
            if not entity_name:
                continue
            
            similarity = self.name_matcher.calculate_similarity(name, entity_name)
            if similarity >= 0.8:  # Higher threshold for PEP
                match = ScreeningMatch(
                    list_type=ScreeningListType.PEP_LIST,
                    match_score=similarity,
                    matched_name=entity_name,
                    matched_entity=entity,
                    confidence_level="high" if similarity > 0.95 else "medium"
                )
                matches.append(match)
        
        return matches
    
    async def _ensure_lists_updated(self) -> None:
        """Ensure screening lists are up to date."""
        now = datetime.utcnow()
        update_threshold = timedelta(hours=24)  # Update daily
        
        needs_update = False
        for list_type in ScreeningListType:
            last_update = self.last_update.get(list_type)
            if not last_update or (now - last_update) > update_threshold:
                needs_update = True
                break
        
        if needs_update:
            await self.update_screening_lists()
    
    def _get_lists_version(self) -> str:
        """Get version identifier for current screening lists."""
        timestamps = [
            self.last_update.get(list_type, datetime.min)
            for list_type in ScreeningListType
        ]
        latest = max(timestamps)
        return latest.strftime("%Y%m%d_%H%M%S")