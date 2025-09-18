"""
Compliance repository for database operations.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, insert, update, delete, and_, or_

from src.models.compliance import (
    KYCData, AMLScreeningResult, AuditLog, ComplianceReport,
    SanctionsList, SanctionsEntry, DataRetentionPolicy
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ComplianceRepository:
    """Repository for compliance-related database operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_async_engine(
            self.settings.database_url,
            echo=False,
            pool_size=20,
            max_overflow=30
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def store_kyc_data(self, kyc_data: KYCData) -> bool:
        """Store KYC verification data."""
        try:
            async with self.async_session() as session:
                # Convert to dict for insertion
                data = kyc_data.dict()
                data['kyc_id'] = str(data['kyc_id'])
                data['customer_id'] = str(data['customer_id'])
                
                # Insert into kyc_data table
                query = """
                INSERT INTO kyc_data (
                    kyc_id, customer_id, status, provider, verification_id,
                    document_type, identity_verified, address_verified,
                    document_verified, selfie_verified, confidence_score,
                    extracted_data, verification_timestamp, expiry_date,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                )
                ON CONFLICT (kyc_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    confidence_score = EXCLUDED.confidence_score,
                    extracted_data = EXCLUDED.extracted_data,
                    verification_timestamp = EXCLUDED.verification_timestamp,
                    updated_at = EXCLUDED.updated_at
                """
                
                await session.execute(query, [
                    data['kyc_id'], data['customer_id'], data['status'],
                    data['provider'], data['verification_id'], data['document_type'],
                    data['identity_verified'], data['address_verified'],
                    data['document_verified'], data['selfie_verified'],
                    data['confidence_score'], data['extracted_data'],
                    data['verification_timestamp'], data['expiry_date'],
                    data['created_at'], data['updated_at']
                ])
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store KYC data: {str(e)}")
            return False
    
    async def get_kyc_data(self, customer_id: UUID) -> Optional[KYCData]:
        """Get KYC data for customer."""
        try:
            async with self.async_session() as session:
                query = """
                SELECT * FROM kyc_data 
                WHERE customer_id = $1 
                ORDER BY created_at DESC 
                LIMIT 1
                """
                
                result = await session.execute(query, [str(customer_id)])
                row = result.fetchone()
                
                if row:
                    return KYCData(**dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get KYC data for {customer_id}: {str(e)}")
            return None
    
    async def store_aml_result(self, aml_result: AMLScreeningResult) -> bool:
        """Store AML screening result."""
        try:
            async with self.async_session() as session:
                data = aml_result.dict()
                data['screening_id'] = str(data['screening_id'])
                data['customer_id'] = str(data['customer_id'])
                if data['reviewer_id']:
                    data['reviewer_id'] = str(data['reviewer_id'])
                
                query = """
                INSERT INTO aml_screening_results (
                    screening_id, customer_id, status, matches,
                    screening_timestamp, lists_version, risk_score,
                    manual_review_required, reviewer_id, review_notes,
                    reviewed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                ON CONFLICT (screening_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    matches = EXCLUDED.matches,
                    risk_score = EXCLUDED.risk_score,
                    manual_review_required = EXCLUDED.manual_review_required,
                    reviewer_id = EXCLUDED.reviewer_id,
                    review_notes = EXCLUDED.review_notes,
                    reviewed_at = EXCLUDED.reviewed_at
                """
                
                await session.execute(query, [
                    data['screening_id'], data['customer_id'], data['status'],
                    data['matches'], data['screening_timestamp'], data['lists_version'],
                    data['risk_score'], data['manual_review_required'],
                    data['reviewer_id'], data['review_notes'], data['reviewed_at']
                ])
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store AML result: {str(e)}")
            return False
    
    async def get_aml_results(self, customer_id: UUID) -> List[AMLScreeningResult]:
        """Get AML screening results for customer."""
        try:
            async with self.async_session() as session:
                query = """
                SELECT * FROM aml_screening_results 
                WHERE customer_id = $1 
                ORDER BY screening_timestamp DESC
                """
                
                result = await session.execute(query, [str(customer_id)])
                rows = result.fetchall()
                
                return [AMLScreeningResult(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get AML results for {customer_id}: {str(e)}")
            return []
    
    async def store_audit_log(self, audit_log: AuditLog) -> bool:
        """Store audit log entry."""
        try:
            async with self.async_session() as session:
                data = audit_log.dict()
                data['log_id'] = str(data['log_id'])
                if data['customer_id']:
                    data['customer_id'] = str(data['customer_id'])
                if data['user_id']:
                    data['user_id'] = str(data['user_id'])
                
                query = """
                INSERT INTO audit_logs (
                    log_id, timestamp, customer_id, user_id, session_id,
                    action, resource, result, ip_address, user_agent,
                    request_id, details, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                )
                """
                
                await session.execute(query, [
                    data['log_id'], data['timestamp'], data['customer_id'],
                    data['user_id'], data['session_id'], data['action'],
                    data['resource'], data['result'], data['ip_address'],
                    data['user_agent'], data['request_id'], data['details'],
                    data['metadata']
                ])
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store audit log: {str(e)}")
            return False
    
    async def get_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        customer_id: Optional[UUID] = None,
        action: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Get audit logs with filters."""
        try:
            async with self.async_session() as session:
                conditions = ["timestamp BETWEEN $1 AND $2"]
                params = [start_date, end_date]
                param_count = 2
                
                if customer_id:
                    param_count += 1
                    conditions.append(f"customer_id = ${param_count}")
                    params.append(str(customer_id))
                
                if action:
                    param_count += 1
                    conditions.append(f"action = ${param_count}")
                    params.append(action)
                
                param_count += 1
                where_clause = " AND ".join(conditions)
                
                query = f"""
                SELECT * FROM audit_logs 
                WHERE {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ${param_count}
                """
                params.append(limit)
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                return [AuditLog(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get audit logs: {str(e)}")
            return []
    
    async def store_screening_lists(self, screening_lists: Dict[str, List[Dict]]) -> bool:
        """Store sanctions screening lists."""
        try:
            async with self.async_session() as session:
                for list_type, entries in screening_lists.items():
                    # Store list metadata
                    list_id = str(UUID())
                    list_query = """
                    INSERT INTO sanctions_lists (
                        list_id, list_name, list_type, last_updated,
                        version, record_count
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (list_type) DO UPDATE SET
                        last_updated = EXCLUDED.last_updated,
                        version = EXCLUDED.version,
                        record_count = EXCLUDED.record_count
                    """
                    
                    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    await session.execute(list_query, [
                        list_id, list_type, list_type, datetime.utcnow(),
                        version, len(entries)
                    ])
                    
                    # Clear old entries
                    await session.execute(
                        "DELETE FROM sanctions_entries WHERE list_id IN (SELECT list_id FROM sanctions_lists WHERE list_type = $1)",
                        [list_type]
                    )
                    
                    # Store entries in batches
                    batch_size = 1000
                    for i in range(0, len(entries), batch_size):
                        batch = entries[i:i + batch_size]
                        
                        entry_values = []
                        for entry in batch:
                            entry_id = str(UUID())
                            entry_values.extend([
                                entry_id, list_id, entry.get('uid', ''),
                                entry.get('names', []), entry.get('aliases', []),
                                entry.get('addresses', []), entry.get('date_of_birth'),
                                entry.get('place_of_birth'), entry.get('nationality'),
                                entry.get('programs', []), entry.get('entity_type', 'individual'),
                                entry.get('remarks'), datetime.utcnow()
                            ])
                        
                        if entry_values:
                            placeholders = []
                            for j in range(len(batch)):
                                start_idx = j * 13 + 1
                                placeholder = f"(${start_idx}, ${start_idx+1}, ${start_idx+2}, ${start_idx+3}, ${start_idx+4}, ${start_idx+5}, ${start_idx+6}, ${start_idx+7}, ${start_idx+8}, ${start_idx+9}, ${start_idx+10}, ${start_idx+11}, ${start_idx+12})"
                                placeholders.append(placeholder)
                            
                            entry_query = f"""
                            INSERT INTO sanctions_entries (
                                entry_id, list_id, external_id, names, aliases,
                                addresses, date_of_birth, place_of_birth, nationality,
                                programs, entity_type, remarks, created_at
                            ) VALUES {', '.join(placeholders)}
                            """
                            
                            await session.execute(entry_query, entry_values)
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store screening lists: {str(e)}")
            return False
    
    async def search_sanctions_entries(
        self,
        name: str,
        list_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search sanctions entries by name."""
        try:
            async with self.async_session() as session:
                conditions = ["names @> $1 OR aliases @> $1"]
                params = [[name.lower()]]
                param_count = 1
                
                if list_types:
                    param_count += 1
                    conditions.append(f"sl.list_type = ANY(${param_count})")
                    params.append(list_types)
                
                where_clause = " AND ".join(conditions)
                
                query = f"""
                SELECT se.*, sl.list_type, sl.list_name
                FROM sanctions_entries se
                JOIN sanctions_lists sl ON se.list_id = sl.list_id
                WHERE {where_clause}
                ORDER BY se.created_at DESC
                LIMIT 100
                """
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search sanctions entries: {str(e)}")
            return []
    
    async def store_compliance_report(self, report: ComplianceReport) -> bool:
        """Store compliance report."""
        try:
            async with self.async_session() as session:
                data = report.dict()
                data['report_id'] = str(data['report_id'])
                data['generated_by'] = str(data['generated_by'])
                if data['customer_id']:
                    data['customer_id'] = str(data['customer_id'])
                
                query = """
                INSERT INTO compliance_reports (
                    report_id, report_type, title, description,
                    period_start, period_end, generated_at, generated_by,
                    customer_id, data, file_path, status
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                )
                """
                
                await session.execute(query, [
                    data['report_id'], data['report_type'], data['title'],
                    data['description'], data['period_start'], data['period_end'],
                    data['generated_at'], data['generated_by'], data['customer_id'],
                    data['data'], data['file_path'], data['status']
                ])
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store compliance report: {str(e)}")
            return False
    
    async def get_compliance_reports(
        self,
        report_type: Optional[str] = None,
        customer_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[ComplianceReport]:
        """Get compliance reports with filters."""
        try:
            async with self.async_session() as session:
                conditions = []
                params = []
                param_count = 0
                
                if report_type:
                    param_count += 1
                    conditions.append(f"report_type = ${param_count}")
                    params.append(report_type)
                
                if customer_id:
                    param_count += 1
                    conditions.append(f"customer_id = ${param_count}")
                    params.append(str(customer_id))
                
                param_count += 1
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                query = f"""
                SELECT * FROM compliance_reports 
                WHERE {where_clause}
                ORDER BY generated_at DESC 
                LIMIT ${param_count}
                """
                params.append(limit)
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                return [ComplianceReport(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get compliance reports: {str(e)}")
            return []
    
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data based on retention policies."""
        try:
            async with self.async_session() as session:
                cleanup_stats = {}
                
                # Get retention policies
                policies_query = "SELECT * FROM data_retention_policies WHERE is_active = true"
                result = await session.execute(policies_query)
                policies = result.fetchall()
                
                for policy in policies:
                    data_type = policy['data_type']
                    retention_days = policy['retention_period_days']
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    
                    if data_type == 'audit_logs':
                        delete_query = "DELETE FROM audit_logs WHERE timestamp < $1"
                        result = await session.execute(delete_query, [cutoff_date])
                        cleanup_stats['audit_logs'] = result.rowcount
                    
                    elif data_type == 'kyc_data':
                        # Anonymize instead of delete for KYC data
                        anonymize_query = """
                        UPDATE kyc_data SET 
                            extracted_data = '{}',
                            updated_at = $1
                        WHERE created_at < $2 AND extracted_data != '{}'
                        """
                        result = await session.execute(anonymize_query, [datetime.utcnow(), cutoff_date])
                        cleanup_stats['kyc_data_anonymized'] = result.rowcount
                
                await session.commit()
                return cleanup_stats
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {str(e)}")
            return {}
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()