"""
Monitoring subsystem for Bull Machine trading system.

Provides real-time monitoring and alerting for:
- Archetype correlation tracking
- Diversification health monitoring
- Performance attribution
- Risk decomposition
"""

from .correlation_monitor import CorrelationMonitor

__all__ = ['CorrelationMonitor']
