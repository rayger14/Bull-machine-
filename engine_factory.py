"""
Engine Factory for Bull Machine v1.7.2
Centralized engine initialization with proper config injection
"""

from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

class EngineFactory:
    """Factory for creating properly configured engines"""

    @staticmethod
    def build_engine(engine_type, config):
        """
        Build engine with proper config injection

        Args:
            engine_type: str - Type of engine to build
            config: dict - Configuration for the engine

        Returns:
            Engine instance
        """

        engines = {
            "smc": lambda: SMCEngine(config),
            "wyckoff": lambda: WyckoffEngine(config),
            "hob": lambda: HOBDetector(config),
            "momentum": lambda: MomentumEngine(config)
        }

        if engine_type.lower() not in engines:
            raise ValueError(f"Unknown engine type: {engine_type}")

        try:
            return engines[engine_type.lower()]()
        except Exception as e:
            raise RuntimeError(f"Failed to build {engine_type} engine: {e}")

    @staticmethod
    def build_all_engines(config):
        """Build all engines from domain config"""

        engines = {}

        # Build each engine with proper config section
        engine_configs = {
            "smc": config['domains']['smc'],
            "momentum": config['domains']['momentum'],
            "wyckoff": config['domains'].get('wyckoff', {}),  # Fallback to empty dict
            "hob": config['domains'].get('liquidity', {}).get('hob_detection', config['domains'].get('hob', {}))
        }

        for engine_name, engine_config in engine_configs.items():
            try:
                engines[engine_name] = EngineFactory.build_engine(engine_name, engine_config)
                print(f"✅ {engine_name.upper()} engine initialized")
            except Exception as e:
                print(f"⚠️  {engine_name.upper()} engine failed: {e}")
                engines[engine_name] = None

        return engines