"""
Position Repository

Data access layer for trading positions.
"""

from typing import List, Optional
from abc import ABC, abstractmethod

from src.models.trading_models import Position


class PositionRepository(ABC):
    """Abstract base class for position data access."""
    
    @abstractmethod
    async def get_active_positions(self) -> List[Position]:
        """Get all active positions."""
        pass
    
    @abstractmethod
    async def get_positions_by_customer(self, customer_id: str) -> List[Position]:
        """Get positions for a specific customer."""
        pass
    
    @abstractmethod
    async def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        pass
    
    @abstractmethod
    async def create_position(self, position: Position) -> Position:
        """Create a new position."""
        pass
    
    @abstractmethod
    async def update_position(self, position: Position) -> Position:
        """Update an existing position."""
        pass
    
    @abstractmethod
    async def delete_position(self, position_id: str) -> bool:
        """Delete a position."""
        pass


class InMemoryPositionRepository(PositionRepository):
    """In-memory implementation for testing."""
    
    def __init__(self):
        self.positions: List[Position] = []
    
    async def get_active_positions(self) -> List[Position]:
        return self.positions.copy()
    
    async def get_positions_by_customer(self, customer_id: str) -> List[Position]:
        return [p for p in self.positions if p.customer_id == customer_id]
    
    async def get_position_by_id(self, position_id: str) -> Optional[Position]:
        for position in self.positions:
            if position.position_id == position_id:
                return position
        return None
    
    async def create_position(self, position: Position) -> Position:
        self.positions.append(position)
        return position
    
    async def update_position(self, position: Position) -> Position:
        for i, p in enumerate(self.positions):
            if p.position_id == position.position_id:
                self.positions[i] = position
                return position
        raise ValueError(f"Position {position.position_id} not found")
    
    async def delete_position(self, position_id: str) -> bool:
        for i, p in enumerate(self.positions):
            if p.position_id == position_id:
                del self.positions[i]
                return True
        return False