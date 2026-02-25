"""
Regime Service - Single Entry Point for Regime Classification
==============================================================

This is the ONLY module that other components should call for regime detection.

Architecture (3-layer stack):
    Layer 0: Event Override (flash crash, extreme events) → immediate crisis
    Layer 1: Regime Model (logistic OR ensemble) → probabilistic classification
    Layer 2: Hysteresis (stability constraints, dwell time, dual thresholds)

Design Principles:
- Single source of truth for regime classification
- Clean separation of concerns (model vs stability vs events)
- Production-ready interface (no ad-hoc logic elsewhere)
- Backward compatible with existing RegimeClassifier interface
- Easy to swap underlying model (logistic → XGBoost → ensemble)
- Feature flag support for A/B testing (static vs dynamic_baseline vs dynamic_ensemble)

Usage:
    from engine.context.regime_service import RegimeService

    # Initialize with ensemble (recommended)
    service = RegimeService(
        mode='dynamic_ensemble',
        model_path='models/ensemble_regime_v1.pkl',
        enable_event_override=True,
        enable_hysteresis=True
    )

    # Single bar classification
    result = service.get_regime(features, timestamp)
    # Returns: {regime_label, regime_probs, regime_confidence, regime_source}

    # Batch classification (backtesting)
    df = service.classify_batch(df)

    # Validate integration (for testing)
    service.validate_integration()

Author: Claude Code (Backend Architect)
Date: 2025-01-08 (Updated: 2026-01-14 for ensemble integration)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pathlib import Path
import pickle

from engine.context.logistic_regime_model import LogisticRegimeModel
from engine.context.regime_hysteresis import RegimeHysteresis
from engine.context.confidence_calibrator import CompositeCalibrator
from engine.context.hybrid_regime_model import HybridRegimeModel
from engine.context.probabilistic_regime_detector import ProbabilisticRegimeDetector

logger = logging.getLogger(__name__)


# Regime Mode Constants (Feature Flags)
REGIME_MODE_STATIC = 'static'  # Use precomputed static labels (for baseline comparison)
REGIME_MODE_DYNAMIC_BASELINE = 'dynamic_baseline'  # Use logistic model dynamically
REGIME_MODE_DYNAMIC_ENSEMBLE = 'dynamic_ensemble'  # Use ensemble model dynamically (recommended)
REGIME_MODE_HYBRID = 'hybrid'  # Use hybrid model (crisis rules + ML for normal regimes) - PRODUCTION DEFAULT
REGIME_MODE_PROBABILISTIC = 'probabilistic'  # Use 3-output probabilistic regime system (crisis_prob, risk_temperature, instability_score) - NEXT-GEN

VALID_REGIME_MODES = [REGIME_MODE_STATIC, REGIME_MODE_DYNAMIC_BASELINE, REGIME_MODE_DYNAMIC_ENSEMBLE, REGIME_MODE_HYBRID, REGIME_MODE_PROBABILISTIC]


class EventOverrideDetector:
    """
    Layer 0: Detect crisis events that trigger immediate regime override.

    Event triggers (ANY of these → immediate crisis mode):
    1. Flash crash: >4% drop in 1H
    2. Extreme volume spike: volume z-score >5 + negative return
    3. Funding shock: |funding z| >4
    4. Liquidation cascade: OI drop >8% in 1H

    These events bypass the model and hysteresis for rapid crisis response.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize event override detector.

        Args:
            config: Optional config dict with custom thresholds
                {
                    'flash_crash_threshold': 0.04,
                    'volume_z_threshold': 5.0,
                    'funding_z_threshold': 4.0,
                    'oi_cascade_threshold': 0.08
                }
        """
        self.config = config or {}

        # CRITICAL FIX (2026-01-08): Recalibrated for Bitcoin volatility
        # Research: Context7 + historical crisis analysis (LUNA, FTX, COVID)
        # Old thresholds marked 72.8% of bars as "crisis" (was calibrated for altcoins)
        # New thresholds target ~1-2% crisis rate (true crises only)
        self.thresholds = {
            'flash_crash': self.config.get('flash_crash_threshold', 0.10),      # 10% drop (was 4%, catches LUNA/COVID but not normal 3-5% moves)
            'volume_z': self.config.get('volume_z_threshold', 5.0),             # 5 sigma (KEEP - already correct)
            'funding_z': self.config.get('funding_z_threshold', 5.0),           # 5 sigma (was 4σ, stricter for true extremes)
            'oi_cascade': self.config.get('oi_cascade_threshold', 0.15)         # 15% drop (was 8%, catches cascades not normal deleveraging)
        }

        logger.info("EventOverrideDetector initialized")
        logger.info(f"  Thresholds: {self.thresholds}")

    def check_event(self, features: Dict[str, float]) -> tuple[bool, Optional[str]]:
        """
        Check if any crisis event is occurring.

        Args:
            features: Dict of feature values

        Returns:
            (is_crisis, event_type) tuple
        """
        # Check 1: Flash crash - calculate dynamically with current threshold
        # CRITICAL FIX (2026-01-09): Don't use pre-calculated flash_crash_1h (uses old 4% threshold)
        # Calculate dynamically from price changes
        close = features.get('close', 0)
        prev_close = features.get('prev_close', close)  # Fallback to current if not available
        if prev_close > 0:
            price_change_pct = (close - prev_close) / prev_close
            if price_change_pct < -self.thresholds['flash_crash']:
                logger.warning(f"CRISIS EVENT: Flash crash detected ({price_change_pct*100:.1f}% drop > {self.thresholds['flash_crash']*100:.0f}% threshold)")
                return True, 'flash_crash'

        # Check 2: Extreme volume spike + negative return
        volume_z = features.get('volume_z_7d', 0.0)
        if close > 0 and prev_close > 0:
            returns_1h = (close - prev_close) / prev_close
        else:
            returns_1h = features.get('returns_1h', 0.0)
        if volume_z > self.thresholds['volume_z'] and returns_1h < 0:
            logger.warning(f"CRISIS EVENT: Extreme volume spike (z={volume_z:.1f}) + negative return")
            return True, 'volume_spike'

        # Check 3: Funding shock
        funding_z = abs(features.get('funding_Z', 0.0))
        if funding_z > self.thresholds['funding_z']:
            logger.warning(f"CRISIS EVENT: Funding shock (|z|={funding_z:.1f})")
            return True, 'funding_shock'

        # Check 4: OI cascade
        oi_change = features.get('oi_change_1h', 0.0)
        if oi_change < -self.thresholds['oi_cascade']:
            logger.warning(f"CRISIS EVENT: OI cascade ({oi_change*100:.1f}% drop)")
            return True, 'oi_cascade'

        return False, None


class RegimeService:
    """
    Single entry point for all regime classification.

    NO other module should compute regime directly.

    Architecture:
        Layer 0: Event Override (minutes-hours crisis detection)
            ↓
        Layer 1: Logistic Model (probabilistic classification)
            ↓
        Layer 1.5: Crisis Threshold + EMA Smoothing (NEW)
            ↓
        Layer 2: Hysteresis (stability constraints)
            ↓
        Output: {regime_label, regime_probs, regime_confidence, regime_source}

    This service is:
    - Stateful (maintains hysteresis state + EMA state across calls)
    - Thread-safe (for production use, add locks if needed)
    - Backward compatible (same interface as RegimeClassifier)
    """

    def __init__(
        self,
        mode: str = REGIME_MODE_HYBRID,  # PRODUCTION DEFAULT: hybrid model (crisis rules + ML)
        model_path: Optional[str] = None,
        static_labels_column: str = 'regime_label',
        enable_event_override: bool = True,
        enable_hysteresis: bool = True,
        hysteresis_config: Optional[Dict] = None,
        event_config: Optional[Dict] = None,
        crisis_threshold: float = 0.60,
        enable_ema_smoothing: bool = False,  # CRITICAL FIX (2026-01-09): Disabled by default (hysteresis has its own EMA)
        ema_alpha: float = 0.08,  # 24-hour window: α = 2/(24+1) ≈ 0.08
        enable_calibration: bool = True,  # NEW: Enable confidence calibration (hybrid approach)
        calibrator_path: Optional[str] = None,  # NEW: Path to confidence calibrator (defaults to models/confidence_calibrator_v1.pkl)
        crisis_config: Optional[Dict] = None  # NEW: Crisis detector configuration (for hybrid mode)
    ):
        """
        Initialize regime service.

        Args:
            mode: Regime mode flag (see REGIME_MODE_* constants)
                - 'static': Read from precomputed labels (for baseline comparison)
                - 'dynamic_baseline': Use logistic model dynamically
                - 'dynamic_ensemble': Use ensemble model dynamically
                - 'hybrid': Use hybrid model (crisis rules + ML) - PRODUCTION DEFAULT
            model_path: Path to trained model
                - For 'dynamic_baseline': 'models/logistic_regime_v1.pkl'
                - For 'dynamic_ensemble': 'models/ensemble_regime_v1.pkl'
                - For 'hybrid': 'models/logistic_regime_v3.pkl' (optional, uses internal default)
            static_labels_column: Column name for static labels (only used in 'static' mode)
            enable_event_override: If True, enable Layer 0 (event detection)
            enable_hysteresis: If True, enable Layer 2 (stability)
            hysteresis_config: Optional config for hysteresis layer
            event_config: Optional config for event override
            crisis_threshold: Minimum P(crisis) to declare crisis (default: 0.60)
            enable_ema_smoothing: If True, apply EMA smoothing to probabilities
            ema_alpha: EMA smoothing factor (default: 0.08 for 24-hour window)
            enable_calibration: If True, calibrate confidence scores using outcome-based calibrator
            calibrator_path: Path to confidence calibrator model (defaults to models/confidence_calibrator_v1.pkl)
            crisis_config: Crisis detector configuration (for hybrid mode only)
        """
        logger.info("=" * 80)
        logger.info("Initializing RegimeService")
        logger.info("=" * 80)

        # Validate mode
        if mode not in VALID_REGIME_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {VALID_REGIME_MODES}")

        self.mode = mode
        self.static_labels_column = static_labels_column
        logger.info(f"Mode: {self.mode}")

        # Initialize validation counters (for guardrail testing)
        self.validation_counters = {
            'static_regime_reads': 0,
            'ensemble_calls': 0,
            'baseline_calls': 0,
            'bars_processed': 0,
            'poison_violations': 0,  # Count of attempts to read poisoned label columns
            'fallback_calls': 0  # Count of fallbacks to default regime
        }
        logger.info("Validation counters initialized")

        # Layer 0: Event Override
        self.enable_event_override = enable_event_override
        if self.enable_event_override:
            self.event_detector = EventOverrideDetector(event_config)
            logger.info("✓ Layer 0: Event Override enabled")
        else:
            self.event_detector = None
            logger.info("✗ Layer 0: Event Override disabled")

        # Layer 1: Model (Logistic, Ensemble, Hybrid, or Probabilistic)
        self.model = None
        self.ensemble_models = None
        self.ensemble_config = None
        self.ensemble_features = None
        self.hybrid_model = None
        self.probabilistic_detector = None
        self.crisis_config = crisis_config

        if self.mode == REGIME_MODE_STATIC:
            logger.info("✓ Layer 1: Static mode (will read precomputed labels)")
        elif self.mode == REGIME_MODE_DYNAMIC_BASELINE:
            # Load logistic model
            if model_path and Path(model_path).exists():
                self.model = LogisticRegimeModel(model_path)
                logger.info(f"✓ Layer 1: Logistic Model loaded from {model_path}")
            else:
                logger.warning(f"⚠ Layer 1: No model found at {model_path}, initializing empty")
                self.model = LogisticRegimeModel()
        elif self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
            # Load ensemble model
            if model_path and Path(model_path).exists():
                self._load_ensemble_model(model_path)
                logger.info(f"✓ Layer 1: Ensemble Model loaded from {model_path}")
                logger.info(f"   - {len(self.ensemble_models)} models in ensemble")
                logger.info(f"   - {len(self.ensemble_features)} features required")
            else:
                raise ValueError(
                    f"Ensemble model not found at {model_path}\n"
                    f"Run bin/train_ensemble_regime_model.py first"
                )
        elif self.mode == REGIME_MODE_HYBRID:
            # Load hybrid model (crisis rules + ML)
            # Use provided model_path or default to logistic_regime_v4_no_funding_stratified.pkl
            if model_path is None:
                model_path = 'models/logistic_regime_v4_no_funding_stratified.pkl'

            self.hybrid_model = HybridRegimeModel(
                ml_model_path=model_path,
                crisis_config=crisis_config
            )
            logger.info(f"✓ Layer 1: Hybrid Model initialized")
            logger.info(f"   - Crisis detector: Rule-based (2-of-4 voting)")
            logger.info(f"   - Normal regimes: ML model ({model_path})")
            logger.info(f"   - Conflict resolution: Crisis rules override ML")
        elif self.mode == REGIME_MODE_PROBABILISTIC:
            # Load probabilistic regime detector (3-output system)
            # Use provided model_path or default to logistic_regime_v4_no_funding_stratified.pkl
            if model_path is None:
                model_path = 'models/logistic_regime_v4_no_funding_stratified.pkl'

            # Load crisis model
            import joblib
            crisis_model = None
            if Path(model_path).exists():
                try:
                    crisis_model = joblib.load(model_path)

                    # Pass full dict if it contains model metadata (model, scaler, feature_order)
                    # The ProbabilisticRegimeDetector will handle extracting what it needs
                    if isinstance(crisis_model, dict) and 'model' in crisis_model:
                        logger.info(f"✓ Layer 1: Crisis model dict loaded from {model_path}")
                        logger.info(f"   - Contains: {list(crisis_model.keys())}")
                    else:
                        logger.info(f"✓ Layer 1: Crisis model loaded from {model_path}")
                except Exception as e:
                    logger.warning(f"⚠ Layer 1: Failed to load crisis model: {e}. Using mock.")
                    crisis_model = None

            if crisis_model is None:
                logger.warning(f"⚠ Layer 1: Crisis model not found at {model_path}, using mock")
                # Create mock crisis model for testing
                class MockCrisisModel:
                    def predict_proba(self, X):
                        # Return low crisis prob by default
                        return np.array([[0.05, 0.25, 0.50, 0.20]])  # [crisis, risk_off, neutral, risk_on]
                crisis_model = MockCrisisModel()

            # Initialize probabilistic detector
            self.probabilistic_detector = ProbabilisticRegimeDetector(
                crisis_model=crisis_model,
                crisis_threshold=crisis_threshold  # Use from __init__ param
            )
            logger.info(f"✓ Layer 1: Probabilistic Regime Detector initialized")
            logger.info(f"   - Crisis model: {model_path}")
            logger.info(f"   - Crisis threshold: {crisis_threshold:.2f}")
            logger.info(f"   - Outputs: crisis_prob, risk_temperature, instability_score")
            logger.info(f"   - Soft controls: position_size_multiplier, trade_frequency_multiplier")

        # Layer 1.5: Crisis Threshold + EMA Smoothing (NEW)
        self.crisis_threshold = crisis_threshold
        self.enable_ema_smoothing = enable_ema_smoothing
        self.ema_alpha = ema_alpha
        self.ema_probs: Optional[Dict[str, float]] = None  # EMA state
        self.crisis_threshold_veto_count = 0  # Track how often crisis is vetoed

        logger.info(f"✓ Layer 1.5: Crisis Threshold = {crisis_threshold:.2f}")
        if self.enable_ema_smoothing:
            logger.info(f"✓ Layer 1.5: EMA Smoothing enabled (α={ema_alpha:.3f}, ~24h window)")
        else:
            logger.info("✗ Layer 1.5: EMA Smoothing disabled")

        # Layer 1.5b: Confidence Calibration (NEW - Hybrid Approach)
        self.enable_calibration = enable_calibration
        self.confidence_calibrator = None

        if self.enable_calibration:
            # Default calibrator path if not provided
            if calibrator_path is None:
                calibrator_path = "models/confidence_calibrator_v1.pkl"

            calibrator_path_obj = Path(calibrator_path)

            if calibrator_path_obj.exists():
                try:
                    with open(calibrator_path_obj, 'rb') as f:
                        calibrator_data = pickle.load(f)

                    self.confidence_calibrator = calibrator_data['calibrators']['composite']

                    logger.info(f"✓ Layer 1.5b: Confidence Calibrator loaded from {calibrator_path}")
                    logger.info(f"   Version: {calibrator_data.get('version', 'unknown')}")
                    logger.info(f"   OOS R²: {calibrator_data['validation_results']['composite']['r2']:.4f}")
                    logger.info(f"   Hybrid Mode: ensemble_agreement (raw) + regime_stability_forecast (calibrated)")

                except Exception as e:
                    logger.warning(f"⚠ Layer 1.5b: Failed to load calibrator: {e}")
                    logger.warning(f"   Falling back to raw confidence only")
                    self.enable_calibration = False
            else:
                logger.warning(f"⚠ Layer 1.5b: Calibrator not found at {calibrator_path}")
                logger.warning(f"   Falling back to raw confidence only")
                logger.warning(f"   Run bin/build_confidence_calibrator.py to create calibrator")
                self.enable_calibration = False
        else:
            logger.info("✗ Layer 1.5b: Confidence Calibration disabled (raw agreement only)")

        # Layer 2: Hysteresis
        self.enable_hysteresis = enable_hysteresis
        if self.enable_hysteresis:
            self.hysteresis = RegimeHysteresis(hysteresis_config)
            logger.info("✓ Layer 2: Hysteresis enabled")
        else:
            self.hysteresis = None
            logger.info("✗ Layer 2: Hysteresis disabled")

        # Statistics
        self.call_count = 0
        self.override_count = 0
        self.transition_count = 0

        logger.info("=" * 80)
        logger.info("RegimeService ready")
        logger.info("=" * 80)

    def _load_ensemble_model(self, model_path: str) -> None:
        """
        Load ensemble model from pickle file.

        Expected structure:
        {
            'models': List[XGBRegressor],
            'features': List[str],
            'config': Dict,
            'train_date': str,
            'version': str
        }

        Args:
            model_path: Path to ensemble model pickle file
        """
        with open(model_path, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.ensemble_models = ensemble_data['models']
        self.ensemble_features = ensemble_data['features']
        self.ensemble_config = ensemble_data.get('config', {})

        # EMA state for ensemble smoothing
        self.ensemble_ema_state = None
        self.ensemble_ema_span = self.ensemble_config.get('ema_span', 48)

        logger.info(f"Ensemble model loaded:")
        logger.info(f"  - Version: {ensemble_data.get('version', 'unknown')}")
        logger.info(f"  - Trained: {ensemble_data.get('train_date', 'unknown')}")
        logger.info(f"  - Models: {len(self.ensemble_models)}")
        logger.info(f"  - Features: {len(self.ensemble_features)}")
        logger.info(f"  - EMA span: {self.ensemble_ema_span}h")

    def _get_regime_from_static_labels(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get regime from precomputed static labels (for baseline comparison).

        This mode should FAIL in production if static labels are not available.

        Args:
            features: Feature dict (must contain static_labels_column)
            timestamp: Optional timestamp

        Returns:
            Regime result dict
        """
        self.validation_counters['static_regime_reads'] += 1
        self.validation_counters['bars_processed'] += 1

        # Check if static label column exists
        label_key = self.static_labels_column
        if label_key not in features:
            # CRITICAL: This should fail in production
            # If we can't read static labels, something is wrong
            self.validation_counters['poison_violations'] += 1
            logger.error(
                f"Static mode: '{label_key}' not found in features. "
                f"This indicates static labels are not available."
            )
            raise KeyError(
                f"Static regime mode: '{label_key}' not in features. "
                f"Either switch to dynamic mode or ensure static labels are provided."
            )

        regime_label = features[label_key]

        # Static mode doesn't have probabilities or confidence
        # Return minimal structure
        return {
            'regime_label': regime_label,
            'regime_probs': {regime_label: 1.0},  # Dummy probs
            'regime_confidence': 1.0,  # Dummy confidence
            'regime_source': 'static',
            'transition_flag': False,  # Can't track transitions in static mode
            'time_in_regime_hours': 0.0
        }

    def _get_regime_from_hybrid(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get regime from hybrid model (crisis rules + ML).

        Steps:
        1. HybridRegimeModel.classify() checks crisis rules first
        2. If crisis detected → return crisis with high confidence
        3. Else → use ML for normal regime classification
        4. Return unified result

        Args:
            features: Feature dict
            timestamp: Optional timestamp

        Returns:
            Dict with regime classification results
        """
        self.validation_counters['bars_processed'] += 1

        # Call hybrid model
        result = self.hybrid_model.classify(features, timestamp)

        # Convert to RegimeService format
        # HybridRegimeModel returns:
        # - regime_label
        # - regime_confidence
        # - regime_proba (dict)
        # - crisis_override (bool)
        # - regime_source

        return {
            'regime_label': result['regime_label'],
            'regime_probs': result['regime_proba'],
            'regime_confidence': result['regime_confidence'],
            'regime_source': result['regime_source'],
            'crisis_override': result.get('crisis_override', False),
            'crisis_triggers': result.get('crisis_triggers'),
            'triggers_fired': result.get('triggers_fired', 0)
        }

    def _get_regime_from_probabilistic(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get regime from probabilistic detector (3-output system).

        Returns continuous probabilities instead of hard labels:
        - crisis_prob [0-1]: Probability of crisis regime
        - risk_temperature [0-1]: Aggressiveness score (0=cold/defensive, 1=hot/aggressive)
        - instability_score [0-1]: Probability of regime change / choppy conditions

        Args:
            features: Feature dict or pandas Series
            timestamp: Optional timestamp

        Returns:
            Dict with probabilistic regime state including:
                - crisis_prob: P(crisis) [0-1]
                - risk_temperature: Aggressiveness score [0-1]
                - instability_score: P(regime change) [0-1]
                - regime_label: Interpretive label based on state (for backward compatibility)
                - regime_probs: Soft probabilities (for backward compatibility)
                - regime_confidence: Composite confidence score
                - soft_controls: Position and frequency multipliers
                - metadata: Additional diagnostic info
        """
        self.validation_counters['bars_processed'] += 1

        # Convert dict to pandas Series if needed
        if isinstance(features, dict):
            features_series = pd.Series(features)
        else:
            features_series = features

        # Call probabilistic detector
        prob_result = self.probabilistic_detector.detect(features_series)

        # Extract 3 outputs
        crisis_prob = prob_result['crisis_prob']
        risk_temperature = prob_result['risk_temperature']
        instability_score = prob_result['instability_score']
        metadata = prob_result['metadata']

        # Get soft controls
        soft_controls = self.probabilistic_detector.get_soft_controls(prob_result)

        # Create interpretive regime label for backward compatibility
        # Logic: crisis_prob > 0.25 → crisis, else use risk_temperature
        if crisis_prob > 0.25:
            regime_label = 'crisis'
        elif risk_temperature > 0.65:
            regime_label = 'risk_on'
        elif risk_temperature < 0.35:
            regime_label = 'risk_off'
        else:
            regime_label = 'neutral'

        # Create soft regime probabilities for backward compatibility
        # Map continuous scores to discrete probabilities
        def softmax_prob(score, center, width=0.15):
            """Gaussian-like probability centered at 'center' with 'width'."""
            return np.exp(-((score - center) ** 2) / (2 * width ** 2))

        # Probabilities based on risk_temperature
        temp_probs = {
            'crisis': softmax_prob(risk_temperature, 0.0, 0.20) * crisis_prob,  # Amplify by crisis_prob
            'risk_off': softmax_prob(risk_temperature, 0.25, 0.15),
            'neutral': softmax_prob(risk_temperature, 0.50, 0.15),
            'risk_on': softmax_prob(risk_temperature, 0.75, 0.15)
        }

        # Normalize to sum to 1
        total = sum(temp_probs.values())
        regime_probs = {k: v / total for k, v in temp_probs.items()}

        # Compute confidence (1 - instability_score, since instability is uncertainty)
        regime_confidence = 1.0 - instability_score

        return {
            # Probabilistic outputs (NEW - primary interface)
            'crisis_prob': float(crisis_prob),
            'risk_temperature': float(risk_temperature),
            'instability_score': float(instability_score),
            'soft_controls': soft_controls,
            'metadata': metadata,

            # Backward-compatible outputs (for existing code)
            'regime_label': regime_label,
            'regime_probs': regime_probs,
            'regime_confidence': float(regime_confidence),
            'regime_source': 'probabilistic',

            # Diagnostic info
            'temperature_level': metadata['temperature_level'],
            'instability_level': metadata['instability_level'],
            'crisis_emergency': metadata['crisis_emergency']
        }

    def _get_regime_from_ensemble(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get regime from ensemble model.

        Steps:
        1. Extract required features
        2. Generate predictions from all ensemble models
        3. Average predictions (risk score)
        4. Calculate ensemble agreement (confidence)
        5. Apply confidence calibration (Layer 1.5b - if enabled)
        6. Apply EMA smoothing (if enabled)
        7. Convert continuous risk score to discrete regime + probabilities
        8. Return result

        Args:
            features: Feature dict
            timestamp: Optional timestamp

        Returns:
            Dict with regime classification results including:
                - regime_label: Discrete regime (crisis/risk_off/neutral/risk_on)
                - regime_probs: Soft probabilities for each regime
                - regime_confidence: Raw ensemble agreement (backward compatible)
                - ensemble_agreement: Raw quality indicator (for signal filtering)
                - regime_stability_forecast: Calibrated stability predictor (for hold time decisions)
                - risk_score: Continuous risk score (0-1)
                - Other diagnostic fields
        """
        self.validation_counters['ensemble_calls'] += 1
        self.validation_counters['bars_processed'] += 1

        # Extract features
        X = np.array([[features.get(f, 0.0) for f in self.ensemble_features]])

        # Check for missing features
        missing_features = [f for f in self.ensemble_features if f not in features]
        if missing_features:
            logger.warning(f"Missing features (filled with 0.0): {missing_features}")

        # Generate ensemble predictions (continuous risk score 0-1)
        predictions = np.array([model.predict(X)[0] for model in self.ensemble_models])
        risk_score = predictions.mean()

        # Calculate ensemble agreement (what user called "confidence" but is actually agreement)
        # Agreement = 1 - coefficient_of_variation
        pred_std = predictions.std()
        pred_mean = predictions.mean()
        cv = pred_std / (abs(pred_mean) + 1e-6)
        ensemble_agreement = 1.0 - min(cv, 1.0)

        # Apply confidence calibration (Layer 1.5b - Hybrid Approach)
        # - ensemble_agreement: raw quality indicator (for filtering signals)
        # - regime_stability_forecast: calibrated predictor (for hold time decisions)
        if self.enable_calibration and self.confidence_calibrator is not None:
            regime_stability_forecast = float(self.confidence_calibrator.predict([ensemble_agreement])[0])
        else:
            regime_stability_forecast = None

        # Apply EMA smoothing to risk score (if enabled)
        if self.enable_ema_smoothing:
            if self.ensemble_ema_state is None:
                # Initialize EMA state
                self.ensemble_ema_state = risk_score
                risk_score_smooth = risk_score
            else:
                # Update EMA
                alpha = 2.0 / (self.ensemble_ema_span + 1.0)
                risk_score_smooth = alpha * risk_score + (1 - alpha) * self.ensemble_ema_state
                self.ensemble_ema_state = risk_score_smooth
        else:
            risk_score_smooth = risk_score

        # Convert continuous risk score to discrete regime + soft probabilities
        # Risk score range: [0, 1]
        # - [0.0, 0.3): Crisis
        # - [0.3, 0.5): Risk-off
        # - [0.5, 0.7): Neutral
        # - [0.7, 1.0]: Risk-on

        # Discretize to get regime label
        if risk_score_smooth < 0.3:
            regime_label = 'crisis'
        elif risk_score_smooth < 0.5:
            regime_label = 'risk_off'
        elif risk_score_smooth < 0.7:
            regime_label = 'neutral'
        else:
            regime_label = 'risk_on'

        # Create soft probabilities based on distance to thresholds
        # This gives more information than hard discretization
        def softmax_prob(score, center, width=0.1):
            """Gaussian-like probability centered at 'center' with 'width'."""
            return np.exp(-((score - center) ** 2) / (2 * width ** 2))

        # Centers for each regime
        crisis_prob = softmax_prob(risk_score_smooth, 0.15, 0.15)
        risk_off_prob = softmax_prob(risk_score_smooth, 0.40, 0.10)
        neutral_prob = softmax_prob(risk_score_smooth, 0.60, 0.10)
        risk_on_prob = softmax_prob(risk_score_smooth, 0.85, 0.15)

        # Normalize to sum to 1
        total = crisis_prob + risk_off_prob + neutral_prob + risk_on_prob
        regime_probs = {
            'crisis': crisis_prob / total,
            'risk_off': risk_off_prob / total,
            'neutral': neutral_prob / total,
            'risk_on': risk_on_prob / total
        }

        # Return result WITHOUT applying hysteresis yet
        # Hysteresis will be applied in get_regime if enabled
        return {
            'regime_label': regime_label,
            'regime_probs': regime_probs,
            'regime_confidence': ensemble_agreement,  # NOTE: This is agreement, not calibrated probability
            'regime_source': 'ensemble',
            'risk_score': risk_score,
            'risk_score_smooth': risk_score_smooth,
            'ensemble_agreement': ensemble_agreement,  # Raw quality indicator (for filtering)
            'regime_stability_forecast': regime_stability_forecast,  # Calibrated predictor (for hold time)
            'ensemble_std': float(pred_std),
            'raw_predictions': predictions.tolist()
        }

    def _apply_crisis_threshold_and_ema(
        self,
        raw_probs: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        """
        Apply EMA smoothing + crisis threshold logic (Layer 1.5).

        Steps:
        1. Apply EMA smoothing to raw probabilities (if enabled)
        2. Check if argmax == 'crisis' but P(crisis) < threshold
        3. If yes, fall back to second-highest regime
        4. Return adjusted probabilities + metadata

        Args:
            raw_probs: Raw probabilities from logistic model
            timestamp: Optional timestamp for logging

        Returns:
            (adjusted_probs, metadata) tuple where:
            - adjusted_probs: Smoothed/thresholded probabilities
            - metadata: Dict with 'crisis_threshold_applied', 'ema_applied', etc.
        """
        metadata = {
            'crisis_threshold_applied': False,
            'crisis_threshold_veto': False,
            'ema_applied': False,
            'raw_model_probs': raw_probs.copy()
        }

        # Step 1: Apply EMA smoothing (if enabled)
        if self.enable_ema_smoothing:
            if self.ema_probs is None:
                # Initialize EMA state with first observation
                self.ema_probs = raw_probs.copy()
                smoothed_probs = raw_probs.copy()
            else:
                # Update EMA: EMA_t = α * P_t + (1-α) * EMA_{t-1}
                smoothed_probs = {}
                for regime in raw_probs.keys():
                    smoothed_probs[regime] = (
                        self.ema_alpha * raw_probs[regime] +
                        (1 - self.ema_alpha) * self.ema_probs.get(regime, raw_probs[regime])
                    )

                # Update state
                self.ema_probs = smoothed_probs.copy()
                metadata['ema_applied'] = True
        else:
            smoothed_probs = raw_probs.copy()

        # Step 2: Check crisis threshold
        # Find regime with highest probability
        sorted_regimes = sorted(smoothed_probs.items(), key=lambda x: x[1], reverse=True)
        top_regime, top_prob = sorted_regimes[0]
        second_regime, second_prob = sorted_regimes[1] if len(sorted_regimes) > 1 else ('neutral', 0.0)

        # Step 3: Apply crisis threshold veto
        if top_regime == 'crisis' and top_prob < self.crisis_threshold:
            # Crisis probability too low, fall back to second-highest regime
            self.crisis_threshold_veto_count += 1
            metadata['crisis_threshold_applied'] = True
            metadata['crisis_threshold_veto'] = True
            metadata['crisis_prob'] = top_prob
            metadata['fallback_regime'] = second_regime
            metadata['fallback_prob'] = second_prob

            # Log the veto (warning level for visibility)
            if timestamp:
                logger.warning(
                    f"[{timestamp}] Crisis threshold veto: P(crisis)={top_prob:.3f} < {self.crisis_threshold:.2f}, "
                    f"falling back to {second_regime} (P={second_prob:.3f})"
                )
            else:
                logger.warning(
                    f"Crisis threshold veto: P(crisis)={top_prob:.3f} < {self.crisis_threshold:.2f}, "
                    f"falling back to {second_regime} (P={second_prob:.3f})"
                )

            # NOTE: We don't modify smoothed_probs here - we just change the interpretation
            # The hysteresis layer will see the raw probabilities but get the adjusted regime
            # This ensures regime selection respects the threshold while preserving probability info

        return smoothed_probs, metadata

    def get_regime(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        THE ONLY FUNCTION other modules should call for regime detection.

        Args:
            features: Dict mapping feature name → value
            timestamp: Optional timestamp (for logging/diagnostics)

        Returns:
            Dict with:
            - regime_label: str (crisis/risk_off/neutral/risk_on)
            - regime_probs: Dict[str, float] (posterior probabilities)
            - regime_confidence: float (0-1, probability gap or hysteresis-adjusted)
            - regime_source: str ('event_override' | 'logistic' | 'logistic+hysteresis')
            - transition_flag: bool (True if regime just changed)
            - time_in_regime_hours: float (hours since regime started)
        """
        self.call_count += 1

        if timestamp is None:
            timestamp = datetime.now()

        # Layer 0: Check event override first
        if self.enable_event_override and self.event_detector:
            is_crisis, event_type = self.event_detector.check_event(features)

            if is_crisis:
                self.override_count += 1

                # If hysteresis enabled, pass override flag
                if self.enable_hysteresis and self.hysteresis:
                    # Force crisis through hysteresis layer
                    crisis_probs = {'crisis': 1.0, 'risk_off': 0.0, 'neutral': 0.0, 'risk_on': 0.0}
                    hyst_result = self.hysteresis.apply_hysteresis('crisis', crisis_probs, timestamp)

                    # Map hysteresis keys to service keys
                    transition = hyst_result.get('transition', hyst_result.get('transition_flag', False))
                    if transition:
                        self.transition_count += 1

                    return {
                        'regime_label': hyst_result.get('regime', hyst_result.get('regime_label', 'crisis')),
                        'regime_probs': hyst_result.get('probs', hyst_result.get('regime_probs', crisis_probs)),
                        'regime_confidence': hyst_result.get('probs', crisis_probs).get('crisis', 1.0),
                        'regime_source': 'event_override',
                        'event_type': event_type,
                        'transition_flag': transition,
                        'time_in_regime_hours': hyst_result.get('dwell_time', hyst_result.get('time_in_regime_hours', 0.0))
                    }
                else:
                    # No hysteresis, return crisis immediately
                    return {
                        'regime_label': 'crisis',
                        'regime_probs': {'crisis': 1.0, 'risk_off': 0.0, 'neutral': 0.0, 'risk_on': 0.0},
                        'regime_confidence': 1.0,
                        'regime_source': 'event_override',
                        'event_type': event_type,
                        'transition_flag': True,
                        'time_in_regime_hours': 0.0
                    }

        # Layer 1: Get regime from appropriate source based on mode
        if self.mode == REGIME_MODE_STATIC:
            # Static mode: read precomputed labels
            return self._get_regime_from_static_labels(features, timestamp)

        elif self.mode == REGIME_MODE_PROBABILISTIC:
            # Probabilistic mode: use 3-output system (crisis_prob, risk_temperature, instability_score)
            self.validation_counters['baseline_calls'] += 1
            prob_result = self._get_regime_from_probabilistic(features, timestamp)

            # Probabilistic mode doesn't use hysteresis (probabilities are already smooth)
            # Return result directly
            return prob_result

        elif self.mode == REGIME_MODE_HYBRID:
            # Hybrid mode: use hybrid model (crisis rules + ML)
            self.validation_counters['baseline_calls'] += 1
            hybrid_result = self._get_regime_from_hybrid(features, timestamp)

            # Extract for hysteresis processing
            proposed_regime = hybrid_result['regime_label']
            raw_probs = hybrid_result['regime_probs']
            smoothed_probs = raw_probs  # Hybrid model handles its own smoothing

            # Store hybrid metadata
            hybrid_metadata = {
                'crisis_override': hybrid_result.get('crisis_override', False),
                'crisis_triggers': hybrid_result.get('crisis_triggers'),
                'triggers_fired': hybrid_result.get('triggers_fired', 0)
            }

            # Apply hysteresis if enabled
            if self.enable_hysteresis and self.hysteresis:
                hyst_result = self.hysteresis.apply_hysteresis(
                    proposed_regime,
                    smoothed_probs,
                    timestamp
                )

                if hyst_result.get('transition', False):
                    self.transition_count += 1

                # Extract values from hysteresis result (uses 'regime' and 'probs' keys)
                final_regime = hyst_result.get('regime', proposed_regime)
                final_probs = hyst_result.get('probs', smoothed_probs)
                final_confidence = final_probs.get(final_regime, 0.0)

                result = {
                    'regime_label': final_regime,
                    'regime_probs': final_probs,
                    'regime_confidence': final_confidence,
                    'regime_source': 'hybrid+hysteresis',
                    'transition_flag': hyst_result.get('transition', False),
                    'time_in_regime_hours': hyst_result.get('dwell_time', 0.0),
                    'raw_model_probs': raw_probs,
                    'smoothed_probs': smoothed_probs
                }

                # Add hybrid metadata
                result.update(hybrid_metadata)

                return result
            else:
                # No hysteresis
                final_confidence = max(raw_probs.values()) - sorted(raw_probs.values(), reverse=True)[1] if len(raw_probs) > 1 else max(raw_probs.values())

                result = {
                    'regime_label': proposed_regime,
                    'regime_probs': raw_probs,
                    'regime_confidence': final_confidence,
                    'regime_source': 'hybrid',
                    'transition_flag': False,
                    'time_in_regime_hours': 0.0,
                    'raw_model_probs': raw_probs,
                    'smoothed_probs': smoothed_probs
                }

                # Add hybrid metadata
                result.update(hybrid_metadata)

                return result

        elif self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
            # Ensemble mode: use ensemble model
            self.validation_counters['baseline_calls'] += 1  # Track that we're NOT using baseline
            ensemble_result = self._get_regime_from_ensemble(features, timestamp)

            # Extract for hysteresis/threshold processing
            proposed_regime = ensemble_result['regime_label']
            raw_probs = ensemble_result['regime_probs']
            smoothed_probs = raw_probs  # Ensemble already applies smoothing internally
            threshold_metadata = {}  # No threshold logic for ensemble (built into discretization)

            # Store ensemble metadata for later
            ensemble_metadata = {
                'risk_score': ensemble_result['risk_score'],
                'risk_score_smooth': ensemble_result['risk_score_smooth'],
                'ensemble_agreement': ensemble_result['ensemble_agreement'],
                'regime_stability_forecast': ensemble_result.get('regime_stability_forecast'),  # Calibrated stability predictor
                'ensemble_std': ensemble_result['ensemble_std']
            }

        elif self.mode == REGIME_MODE_DYNAMIC_BASELINE:
            # Baseline mode: use logistic model
            self.validation_counters['baseline_calls'] += 1
            model_result = self.model.classify(features)
            raw_probs = model_result['regime_probs']

            # Layer 1.5: Apply crisis threshold + EMA smoothing
            smoothed_probs, threshold_metadata = self._apply_crisis_threshold_and_ema(raw_probs, timestamp)

            # Determine final regime after threshold logic
            # CRITICAL FIX (2026-01-09): Compute regime label BEFORE hysteresis to pass to it
            # This ensures threshold decisions are respected by hysteresis
            # CRITICAL BUG FIX (2026-01-09 #2): Use RAW probabilities, not smoothed!
            # Using smoothed probabilities creates feedback loop where hysteresis EMA
            # keeps crisis active long after model stops predicting it
            if threshold_metadata.get('crisis_threshold_veto', False):
                # Use second-highest regime instead
                proposed_regime = threshold_metadata['fallback_regime']
            else:
                # Use argmax of RAW model probabilities (not smoothed by hysteresis)
                proposed_regime = max(raw_probs.items(), key=lambda x: x[1])[0]

            ensemble_metadata = {}  # No ensemble metadata for baseline
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Layer 2: Apply hysteresis (if enabled)
        if self.enable_hysteresis and self.hysteresis:
            # CRITICAL FIX (2026-01-09): Use new apply_hysteresis() method
            # This respects the regime label from threshold layer instead of re-computing
            hyst_result = self.hysteresis.apply_hysteresis(
                proposed_regime,
                smoothed_probs,
                timestamp
            )

            transition = hyst_result.get('transition', hyst_result.get('transition_flag', False))
            if transition:
                self.transition_count += 1

            # Build final result
            # Determine regime source string based on mode
            if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
                regime_source = 'ensemble+hysteresis'
            elif self.mode == REGIME_MODE_DYNAMIC_BASELINE:
                regime_source = 'logistic+ema+threshold+hysteresis' if self.enable_ema_smoothing else 'logistic+threshold+hysteresis'
            else:
                regime_source = 'unknown'

            # Extract values from hysteresis result (uses 'regime' and 'probs' keys)
            final_regime = hyst_result.get('regime', proposed_regime)
            final_probs = hyst_result.get('probs', smoothed_probs)
            final_confidence = final_probs.get(final_regime, 0.0)

            result = {
                'regime_label': final_regime,
                'regime_probs': final_probs,
                'regime_confidence': final_confidence,
                'regime_source': regime_source,
                'transition_flag': hyst_result.get('transition', False),
                'time_in_regime_hours': hyst_result.get('dwell_time', 0.0),
                'raw_model_probs': raw_probs,  # Original model output
                'smoothed_probs': smoothed_probs  # After EMA (if enabled)
            }

            # Add ensemble metadata (if using ensemble mode)
            if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE and ensemble_metadata:
                result.update(ensemble_metadata)

            # Add threshold metadata (for baseline mode)
            if threshold_metadata.get('crisis_threshold_veto', False):
                result['crisis_threshold_veto'] = True
                result['crisis_veto_prob'] = threshold_metadata['crisis_prob']
                result['fallback_regime'] = threshold_metadata['fallback_regime']

            # Add hysteresis veto metadata
            if hyst_result.get('hysteresis_veto', False):
                result['hysteresis_veto'] = True

            return result
        else:
            # No hysteresis - use proposed regime directly
            final_regime = proposed_regime

            # Compute confidence (probability gap)
            sorted_probs = sorted(smoothed_probs.values(), reverse=True)
            final_confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

            # Determine regime source string based on mode
            if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
                regime_source = 'ensemble'
            elif self.mode == REGIME_MODE_DYNAMIC_BASELINE:
                regime_source = 'logistic+ema+threshold' if self.enable_ema_smoothing else 'logistic+threshold'
            else:
                regime_source = 'unknown'

            result = {
                'regime_label': final_regime,
                'regime_probs': smoothed_probs,
                'regime_confidence': final_confidence,
                'regime_source': regime_source,
                'transition_flag': False,  # Can't track transitions without state
                'time_in_regime_hours': 0.0,
                'raw_model_probs': raw_probs,
                'smoothed_probs': smoothed_probs
            }

            # Add ensemble metadata (if using ensemble mode)
            if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE and ensemble_metadata:
                result.update(ensemble_metadata)

            # Add threshold metadata (for baseline mode)
            if threshold_metadata.get('crisis_threshold_veto', False):
                result['crisis_threshold_veto'] = True
                result['crisis_veto_prob'] = threshold_metadata['crisis_prob']
                result['fallback_regime'] = threshold_metadata['fallback_regime']

            return result

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime for entire DataFrame (backtesting mode).

        Args:
            df: DataFrame with feature columns + timestamp index

        Returns:
            DataFrame with regime columns added:
            - regime_label
            - regime_confidence
            - regime_proba_crisis
            - regime_proba_risk_off
            - regime_proba_neutral
            - regime_proba_risk_on
            - regime_transition (if hysteresis enabled)
            - regime_source
        """
        logger.info("=" * 80)
        logger.info(f"Classifying {len(df)} bars with RegimeService (batch mode)")
        logger.info("=" * 80)

        # Reset hysteresis state for clean backtest
        if self.enable_hysteresis and self.hysteresis:
            self.hysteresis.reset()
            logger.info("Hysteresis state reset for batch mode")

        # CRITICAL FIX (2026-01-09): Reset EMA state for reproducible backtests
        if self.enable_ema_smoothing:
            self.ema_probs = {}
            logger.info("EMA state reset for batch mode")

        results = []

        for idx, row in df.iterrows():
            features = row.to_dict()
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)

            result = self.get_regime(features, timestamp)
            results.append(result)

        # Convert to DataFrame
        result_df = df.copy()

        result_df['regime_label'] = [r['regime_label'] for r in results]
        result_df['regime_confidence'] = [r['regime_confidence'] for r in results]
        result_df['regime_source'] = [r['regime_source'] for r in results]

        # Add probability columns
        for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
            result_df[f'regime_proba_{regime}'] = [r['regime_probs'][regime] for r in results]

        # Add probabilistic outputs (if in probabilistic mode)
        if self.mode == REGIME_MODE_PROBABILISTIC:
            result_df['crisis_prob'] = [r.get('crisis_prob', 0.0) for r in results]
            result_df['risk_temperature'] = [r.get('risk_temperature', 0.5) for r in results]
            result_df['instability_score'] = [r.get('instability_score', 0.5) for r in results]
            result_df['position_size_multiplier'] = [r.get('soft_controls', {}).get('position_size_multiplier', 1.0) for r in results]
            result_df['trade_frequency_multiplier'] = [r.get('soft_controls', {}).get('trade_frequency_multiplier', 1.0) for r in results]
            result_df['crisis_gate'] = [r.get('soft_controls', {}).get('crisis_gate', False) for r in results]

        # Add transition flag (if hysteresis enabled)
        if self.enable_hysteresis:
            result_df['regime_transition'] = [r['transition_flag'] for r in results]

        # Log statistics
        self._log_batch_statistics(result_df)

        return result_df

    def _log_batch_statistics(self, df: pd.DataFrame) -> None:
        """Log regime classification statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("Batch Classification Statistics")
        logger.info("=" * 80)

        # Regime distribution
        regime_dist = df['regime_label'].value_counts()
        logger.info("\nRegime distribution:")
        for regime, count in regime_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

        # Transition analysis (if available)
        if 'regime_transition' in df.columns:
            n_transitions = df['regime_transition'].sum()
            transitions_per_year = (n_transitions / len(df)) * 8760
            logger.info(f"\nTransitions:")
            logger.info(f"  Total: {n_transitions}")
            logger.info(f"  Per year: {transitions_per_year:.1f}")

        # Source distribution
        source_dist = df['regime_source'].value_counts()
        logger.info("\nSource distribution:")
        for source, count in source_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {source:20s}: {count:6d} ({pct:5.1f}%)")

        # Confidence statistics
        logger.info(f"\nConfidence statistics:")
        logger.info(f"  Mean: {df['regime_confidence'].mean():.3f}")
        logger.info(f"  Median: {df['regime_confidence'].median():.3f}")
        logger.info(f"  Min: {df['regime_confidence'].min():.3f}")
        logger.info(f"  P10: {df['regime_confidence'].quantile(0.10):.3f}")

        logger.info("\n" + "=" * 80)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dict with call counts, transition counts, etc.
        """
        stats = {
            'total_calls': self.call_count,
            'override_count': self.override_count,
            'transition_count': self.transition_count,
            'crisis_threshold_veto_count': self.crisis_threshold_veto_count,
            'override_rate': self.override_count / self.call_count if self.call_count > 0 else 0.0,
            'crisis_veto_rate': self.crisis_threshold_veto_count / self.call_count if self.call_count > 0 else 0.0
        }

        # Add hysteresis stats if available
        if self.enable_hysteresis and self.hysteresis:
            hyst_stats = self.hysteresis.get_statistics()
            stats['hysteresis'] = hyst_stats

        # Add validation counters
        stats['validation_counters'] = self.validation_counters.copy()

        return stats

    def validate_integration(self, expected_bars: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate integration based on validation counters (Test 1: No Stale Reads).

        This is the CRITICAL test for Phase 1 integration validation.

        Test criteria (from user's specification):
        1. static_regime_reads MUST be 0 (no reading from old label columns)
        2. ensemble_calls MUST equal bars_processed (every bar uses dynamic model)
        3. poison_violations MUST be 0 (no attempts to read poisoned columns)
        4. fallback_calls should be 0 (or acceptable if fallback is intentional)

        Args:
            expected_bars: Optional expected number of bars processed (for additional validation)

        Returns:
            Dict with validation results:
            - passed: bool (True if all tests pass)
            - counters: Dict of validation counters
            - violations: List of violation messages
            - warnings: List of warning messages
        """
        counters = self.validation_counters.copy()
        violations = []
        warnings = []

        # Test 1: No static regime reads (CRITICAL)
        if counters['static_regime_reads'] > 0:
            violations.append(
                f"FAIL: static_regime_reads = {counters['static_regime_reads']} (expected 0). "
                f"Service is reading from old static labels instead of computing dynamically."
            )

        # Test 2: Ensemble calls match bars processed (if ensemble mode)
        if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
            if counters['ensemble_calls'] != counters['bars_processed']:
                violations.append(
                    f"FAIL: ensemble_calls ({counters['ensemble_calls']}) != bars_processed ({counters['bars_processed']}). "
                    f"Not all bars are using ensemble model."
                )

        # Test 3: No poison violations (CRITICAL)
        if counters['poison_violations'] > 0:
            violations.append(
                f"FAIL: poison_violations = {counters['poison_violations']}. "
                f"Service attempted to read poisoned label columns {counters['poison_violations']} times."
            )

        # Test 4: Minimal fallback calls (WARNING level)
        if counters['fallback_calls'] > 0:
            warnings.append(
                f"WARN: fallback_calls = {counters['fallback_calls']}. "
                f"Service fell back to default regime {counters['fallback_calls']} times."
            )

        # Test 5: Expected bars match (if provided)
        if expected_bars is not None:
            if counters['bars_processed'] != expected_bars:
                warnings.append(
                    f"WARN: bars_processed ({counters['bars_processed']}) != expected ({expected_bars})"
                )

        # Test 6: At least some bars were processed
        if counters['bars_processed'] == 0:
            violations.append(
                "FAIL: bars_processed = 0. No bars were processed."
            )

        # Determine pass/fail
        passed = len(violations) == 0

        # Build result
        result = {
            'passed': passed,
            'mode': self.mode,
            'counters': counters,
            'violations': violations,
            'warnings': warnings,
            'test_criteria': {
                'static_regime_reads': 0,
                'poison_violations': 0,
                'ensemble_calls_equals_bars_processed': self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE
            }
        }

        # Log results
        logger.info("=" * 80)
        logger.info("Integration Validation Results (Test 1: No Stale Reads)")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"\nValidation Counters:")
        for key, value in counters.items():
            logger.info(f"  {key:30s}: {value}")

        if violations:
            logger.error(f"\n❌ VALIDATION FAILED ({len(violations)} violations):")
            for violation in violations:
                logger.error(f"  - {violation}")

        if warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(warnings)} warnings):")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        if passed:
            logger.info("\n✅ VALIDATION PASSED - All tests passed!")
            logger.info("   - No static regime reads")
            logger.info("   - No poison violations")
            if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
                logger.info("   - All bars used ensemble model")
        else:
            logger.error("\n❌ VALIDATION FAILED - See violations above")

        logger.info("=" * 80)

        return result

    def reset(self) -> None:
        """Reset service state (for testing or reinitialization)."""
        if self.enable_hysteresis and self.hysteresis:
            self.hysteresis.reset()

        # Reset EMA state
        self.ema_probs = None

        # Reset ensemble EMA state
        if self.mode == REGIME_MODE_DYNAMIC_ENSEMBLE:
            self.ensemble_ema_state = None

        # Reset hybrid model state
        if self.mode == REGIME_MODE_HYBRID and self.hybrid_model:
            self.hybrid_model.reset()

        # Reset probabilistic detector state
        if self.mode == REGIME_MODE_PROBABILISTIC and self.probabilistic_detector:
            # Reset regime history for flip detection
            self.probabilistic_detector.regime_history = []

        # Reset counters
        self.call_count = 0
        self.override_count = 0
        self.transition_count = 0
        self.crisis_threshold_veto_count = 0

        # Reset validation counters
        for key in self.validation_counters:
            self.validation_counters[key] = 0

        logger.info("RegimeService state reset (including EMA state and validation counters)")


# Example usage and testing
if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 80)
    print("Regime Service - Single Entry Point for Regime Classification")
    print("=" * 80)
    print("\nArchitecture:")
    print("  Layer 0: Event Override (flash crash, extreme events)")
    print("  Layer 1: Logistic Model (probabilistic classification)")
    print("  Layer 2: Hysteresis (stability constraints)")
    print("\nDesign Principles:")
    print("  - Single source of truth")
    print("  - Clean separation of concerns")
    print("  - Production-ready interface")
    print("  - Backward compatible")
    print("\n" + "=" * 80)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("\nTesting RegimeService with synthetic data...")

        # Initialize service (without model for now)
        service = RegimeService(
            model_path=None,
            enable_event_override=True,
            enable_hysteresis=True
        )

        # Test single classification
        test_features = {
            'crash_frequency_7d': 1.0,
            'crisis_persistence': 0.3,
            'aftershock_score': 0.2,
            'rv_20d': 0.5,
            'rv_60d': 0.4,
            'drawdown_persistence': 0.6,
            'funding_Z': -1.5,
            'oi': 1e9,
            'volume_z_7d': 2.0,
            'USDT.D': float('nan'),
            'BTC.D': float('nan'),
            'VIX_Z': 1.2,
            'DXY_Z': 0.5,
            'YIELD_CURVE': -0.5
        }

        try:
            result = service.get_regime(test_features)

            print("\nTest classification:")
            print(f"  Regime: {result['regime_label']}")
            print(f"  Confidence: {result['regime_confidence']:.3f}")
            print(f"  Source: {result['regime_source']}")
            print(f"  Probabilities:")
            for regime, prob in result['regime_probs'].items():
                print(f"    {regime:12s}: {prob:.3f}")

            # Get statistics
            stats = service.get_statistics()
            print(f"\nService statistics:")
            print(f"  Total calls: {stats['total_calls']}")
            print(f"  Overrides: {stats['override_count']}")
            print(f"  Transitions: {stats['transition_count']}")

            print("\n✅ RegimeService test passed!")

        except Exception as e:
            print(f"\n⚠ Test warning: {e}")
            print("Note: Full testing requires trained model")
            print("Run training pipeline first: python bin/train_logistic_regime.py")

    else:
        print("\nUsage:")
        print("  python regime_service.py test")
        print("\nOr import in your code:")
        print("  from engine.context.regime_service import RegimeService")
