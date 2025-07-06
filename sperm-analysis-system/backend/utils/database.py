#!/usr/bin/env python3
"""
Database Utilities
Author: Youssef Shitiwi
Description: Database connection and management utilities
"""

import asyncio
from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Float, Integer, Boolean, Text, JSON
from datetime import datetime
from loguru import logger

from backend.utils.config import get_settings

# Database Models
class Base(DeclarativeBase):
    """Base class for database models."""
    pass

class AnalysisRecord(Base):
    """Database model for analysis records."""
    __tablename__ = "analyses"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_name: Mapped[Optional[str]] = mapped_column(String(255))
    video_filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50))
    
    # Configuration
    config: Mapped[dict] = mapped_column(JSON)
    
    # Results summary
    total_sperm_count: Mapped[Optional[int]] = mapped_column(Integer)
    motility_percentage: Mapped[Optional[float]] = mapped_column(Float)
    progressive_percentage: Mapped[Optional[float]] = mapped_column(Float)
    
    # Processing info
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # File paths
    result_files: Mapped[Optional[dict]] = mapped_column(JSON)

class SpermRecord(Base):
    """Database model for individual sperm records."""
    __tablename__ = "sperm_data"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(36), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    
    # Motion classification
    is_motile: Mapped[bool] = mapped_column(Boolean)
    is_progressive: Mapped[bool] = mapped_column(Boolean)
    
    # Velocity parameters
    vcl: Mapped[float] = mapped_column(Float)
    vsl: Mapped[float] = mapped_column(Float)
    vap: Mapped[float] = mapped_column(Float)
    
    # Motion parameters
    lin: Mapped[float] = mapped_column(Float)
    str: Mapped[float] = mapped_column(Float)
    wob: Mapped[float] = mapped_column(Float)
    
    # Path parameters
    alh: Mapped[float] = mapped_column(Float)
    bcf: Mapped[float] = mapped_column(Float)
    
    # Distance parameters
    total_distance: Mapped[float] = mapped_column(Float)
    net_distance: Mapped[float] = mapped_column(Float)
    
    # Duration
    duration_frames: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[float] = mapped_column(Float)
    
    # Trajectory data (optional, stored as JSON)
    trajectory: Mapped[Optional[dict]] = mapped_column(JSON)

# Database Manager
class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection."""
        if not self.settings.use_database or not self.settings.database_url:
            logger.info("Database disabled or URL not provided")
            return
        
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.debug,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        if not self._initialized or not self.session_factory:
            yield None
            return
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    def is_available(self) -> bool:
        """Check if database is available."""
        return self._initialized and self.engine is not None

# Global database manager instance
db_manager = DatabaseManager()

async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async for session in db_manager.get_session():
        yield session

# Database operations
class AnalysisRepository:
    """Repository for analysis operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_analysis(self, analysis_data: dict) -> AnalysisRecord:
        """Create new analysis record."""
        analysis = AnalysisRecord(**analysis_data)
        self.session.add(analysis)
        await self.session.commit()
        await self.session.refresh(analysis)
        return analysis
    
    async def get_analysis(self, analysis_id: str) -> Optional[AnalysisRecord]:
        """Get analysis by ID."""
        from sqlalchemy import select
        
        result = await self.session.execute(
            select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
        )
        return result.scalar_one_or_none()
    
    async def update_analysis(self, analysis_id: str, update_data: dict) -> Optional[AnalysisRecord]:
        """Update analysis record."""
        analysis = await self.get_analysis(analysis_id)
        if not analysis:
            return None
        
        for key, value in update_data.items():
            setattr(analysis, key, value)
        
        await self.session.commit()
        await self.session.refresh(analysis)
        return analysis
    
    async def list_analyses(self, limit: int = 100, offset: int = 0) -> list[AnalysisRecord]:
        """List analyses with pagination."""
        from sqlalchemy import select
        
        result = await self.session.execute(
            select(AnalysisRecord)
            .order_by(AnalysisRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis record."""
        analysis = await self.get_analysis(analysis_id)
        if not analysis:
            return False
        
        await self.session.delete(analysis)
        await self.session.commit()
        return True

class SpermRepository:
    """Repository for sperm data operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_sperm_records(self, sperm_data_list: list[dict]) -> list[SpermRecord]:
        """Create multiple sperm records."""
        sperm_records = [SpermRecord(**data) for data in sperm_data_list]
        self.session.add_all(sperm_records)
        await self.session.commit()
        return sperm_records
    
    async def get_sperm_by_analysis(self, analysis_id: str) -> list[SpermRecord]:
        """Get all sperm data for an analysis."""
        from sqlalchemy import select
        
        result = await self.session.execute(
            select(SpermRecord).where(SpermRecord.analysis_id == analysis_id)
        )
        return list(result.scalars().all())
    
    async def delete_sperm_by_analysis(self, analysis_id: str) -> int:
        """Delete all sperm data for an analysis."""
        from sqlalchemy import delete
        
        result = await self.session.execute(
            delete(SpermRecord).where(SpermRecord.analysis_id == analysis_id)
        )
        await self.session.commit()
        return result.rowcount

# Database service functions
async def save_analysis_to_db(analysis_id: str, analysis_data: dict, sperm_data: list[dict]):
    """Save complete analysis to database."""
    if not db_manager.is_available():
        return
    
    try:
        async for session in db_manager.get_session():
            if session is None:
                return
            
            # Save analysis record
            analysis_repo = AnalysisRepository(session)
            await analysis_repo.create_analysis({
                "id": analysis_id,
                **analysis_data
            })
            
            # Save sperm data
            if sperm_data:
                sperm_repo = SpermRepository(session)
                # Add analysis_id to each sperm record
                for sperm in sperm_data:
                    sperm["analysis_id"] = analysis_id
                
                await sperm_repo.create_sperm_records(sperm_data)
            
            logger.info(f"Analysis {analysis_id} saved to database")
            
    except Exception as e:
        logger.error(f"Failed to save analysis to database: {e}")

async def get_analysis_from_db(analysis_id: str) -> Optional[dict]:
    """Get analysis from database."""
    if not db_manager.is_available():
        return None
    
    try:
        async for session in db_manager.get_session():
            if session is None:
                return None
            
            analysis_repo = AnalysisRepository(session)
            sperm_repo = SpermRepository(session)
            
            # Get analysis record
            analysis = await analysis_repo.get_analysis(analysis_id)
            if not analysis:
                return None
            
            # Get sperm data
            sperm_data = await sperm_repo.get_sperm_by_analysis(analysis_id)
            
            # Convert to dict
            result = {
                "analysis": {
                    "id": analysis.id,
                    "analysis_name": analysis.analysis_name,
                    "video_filename": analysis.video_filename,
                    "status": analysis.status,
                    "config": analysis.config,
                    "total_sperm_count": analysis.total_sperm_count,
                    "motility_percentage": analysis.motility_percentage,
                    "progressive_percentage": analysis.progressive_percentage,
                    "processing_time": analysis.processing_time,
                    "error_message": analysis.error_message,
                    "created_at": analysis.created_at,
                    "started_at": analysis.started_at,
                    "completed_at": analysis.completed_at,
                    "result_files": analysis.result_files
                },
                "sperm_data": [
                    {
                        "track_id": sperm.track_id,
                        "is_motile": sperm.is_motile,
                        "is_progressive": sperm.is_progressive,
                        "vcl": sperm.vcl,
                        "vsl": sperm.vsl,
                        "vap": sperm.vap,
                        "lin": sperm.lin,
                        "str": sperm.str,
                        "wob": sperm.wob,
                        "alh": sperm.alh,
                        "bcf": sperm.bcf,
                        "total_distance": sperm.total_distance,
                        "net_distance": sperm.net_distance,
                        "duration_frames": sperm.duration_frames,
                        "duration_seconds": sperm.duration_seconds,
                        "trajectory": sperm.trajectory
                    }
                    for sperm in sperm_data
                ]
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Failed to get analysis from database: {e}")
        return None

# Initialize database on startup
async def init_database():
    """Initialize database on application startup."""
    await db_manager.initialize()

# Cleanup database on shutdown
async def close_database():
    """Close database connections on application shutdown."""
    await db_manager.close()