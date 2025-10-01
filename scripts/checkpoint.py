#!/usr/bin/env python3
"""
Checkpointing System for Bull Machine v1.7
Enables resumable long-running backtests with state persistence
"""

import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd

class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        """Initialize checkpoint manager"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def create_run_id(self, config, assets, start_date, end_date):
        """Create unique run ID based on parameters"""
        # Create hash of key parameters
        params_str = f"{config}_{assets}_{start_date}_{end_date}"
        run_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"run_{timestamp}_{run_hash}"

    def save_checkpoint(self, run_id, state):
        """Save checkpoint state to disk"""
        checkpoint_file = self.checkpoint_dir / f"{run_id}.checkpoint"

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'run_id': run_id,
            'state': state
        }

        # Save as JSON for human readability
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Also save as pickle for complex objects
        pickle_file = self.checkpoint_dir / f"{run_id}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        return checkpoint_file

    def load_checkpoint(self, run_id):
        """Load checkpoint state from disk"""
        checkpoint_file = self.checkpoint_dir / f"{run_id}.checkpoint"
        pickle_file = self.checkpoint_dir / f"{run_id}.pkl"

        if pickle_file.exists():
            # Prefer pickle for full object restoration
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        elif checkpoint_file.exists():
            # Fallback to JSON
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        else:
            return None

    def checkpoint_exists(self, run_id):
        """Check if checkpoint exists for run"""
        checkpoint_file = self.checkpoint_dir / f"{run_id}.checkpoint"
        pickle_file = self.checkpoint_dir / f"{run_id}.pkl"
        return checkpoint_file.exists() or pickle_file.exists()

    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []

        for file in self.checkpoint_dir.glob("*.checkpoint"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    checkpoints.append({
                        'run_id': data['run_id'],
                        'timestamp': data['timestamp'],
                        'file': file.name
                    })
            except:
                continue

        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)

    def cleanup_old_checkpoints(self, keep_days=7):
        """Clean up checkpoints older than specified days"""
        cutoff = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

        for file in self.checkpoint_dir.glob("*.checkpoint"):
            if file.stat().st_mtime < cutoff:
                file.unlink()

        for file in self.checkpoint_dir.glob("*.pkl"):
            if file.stat().st_mtime < cutoff:
                file.unlink()

class ResumableBacktest:
    """Wrapper for backtests with checkpointing support"""

    def __init__(self, config, assets, start_date, end_date, chunk_days=90):
        """Initialize resumable backtest"""
        self.config = config
        self.assets = assets
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.chunk_days = chunk_days

        self.checkpoint_mgr = CheckpointManager()
        self.run_id = self.checkpoint_mgr.create_run_id(
            str(config), str(assets), str(start_date), str(end_date)
        )

        print(f"🔄 Resumable backtest: {self.run_id}")

    def run(self, resume=True):
        """Run backtest with checkpointing"""
        # Check for existing checkpoint
        if resume and self.checkpoint_mgr.checkpoint_exists(self.run_id):
            print("📁 Loading existing checkpoint...")
            checkpoint = self.checkpoint_mgr.load_checkpoint(self.run_id)
            state = checkpoint['state']
            print(f"   Resuming from chunk {state['current_chunk']}")
        else:
            print("🆕 Starting new backtest...")
            state = self._initialize_state()

        # Generate date chunks if not already done
        if 'chunks' not in state:
            state['chunks'] = self._generate_chunks()

        # Process remaining chunks
        while state['current_chunk'] < len(state['chunks']):
            chunk_start, chunk_end = state['chunks'][state['current_chunk']]

            print(f"📊 Processing chunk {state['current_chunk'] + 1}/{len(state['chunks'])}")
            print(f"   Period: {chunk_start} to {chunk_end}")

            try:
                # Process this chunk
                chunk_result = self._process_chunk(chunk_start, chunk_end, state)

                # Update state
                state['chunk_results'].append(chunk_result)
                state['current_chunk'] += 1
                state['last_processed_date'] = chunk_end

                # Save checkpoint after each chunk
                self.checkpoint_mgr.save_checkpoint(self.run_id, state)

                print(f"   ✅ Chunk completed, checkpoint saved")

            except Exception as e:
                print(f"   ❌ Chunk failed: {e}")
                # Save checkpoint even on failure
                state['errors'].append({
                    'chunk': state['current_chunk'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                self.checkpoint_mgr.save_checkpoint(self.run_id, state)
                raise

        # Aggregate final results
        final_results = self._aggregate_results(state)

        print(f"🏁 Backtest completed: {self.run_id}")
        return final_results

    def _initialize_state(self):
        """Initialize backtest state"""
        return {
            'run_id': self.run_id,
            'config': self.config,
            'assets': self.assets,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'chunk_days': self.chunk_days,
            'current_chunk': 0,
            'chunk_results': [],
            'errors': [],
            'portfolio': {
                'capital': 100000.0,
                'position': 0.0,
                'entry_price': 0.0,
                'trades_count': 0
            },
            'trades': [],
            'last_processed_date': None
        }

    def _generate_chunks(self):
        """Generate date chunks for processing"""
        chunks = []
        current_date = self.start_date

        while current_date < self.end_date:
            chunk_end = min(
                current_date + pd.Timedelta(days=self.chunk_days),
                self.end_date
            )
            chunks.append((current_date, chunk_end))
            current_date = chunk_end

        return chunks

    def _process_chunk(self, chunk_start, chunk_end, state):
        """Process a single date chunk"""
        # This would contain the actual backtest logic
        # For now, return a placeholder result

        import time
        import random

        # Simulate processing time
        time.sleep(random.uniform(1, 3))

        # Simulate chunk results
        return {
            'start_date': chunk_start.isoformat(),
            'end_date': chunk_end.isoformat(),
            'bars_processed': random.randint(50, 200),
            'trades_in_chunk': random.randint(0, 5),
            'chunk_return': random.uniform(-2, 3),
            'processing_time': time.time()
        }

    def _aggregate_results(self, state):
        """Aggregate results from all chunks"""
        total_bars = sum(chunk['bars_processed'] for chunk in state['chunk_results'])
        total_trades = sum(chunk['trades_in_chunk'] for chunk in state['chunk_results'])
        total_return = sum(chunk['chunk_return'] for chunk in state['chunk_results'])

        return {
            'run_id': self.run_id,
            'status': 'completed',
            'total_chunks': len(state['chunks']),
            'total_bars': total_bars,
            'total_trades': total_trades,
            'total_return': total_return,
            'errors': len(state['errors']),
            'chunk_results': state['chunk_results']
        }

def main():
    """Example usage of checkpointing system"""
    print("🔄 RESUMABLE BACKTEST EXAMPLE")
    print("=" * 40)

    # Example config
    config = {
        'version': '1.7.0-test',
        'fusion': {
            'calibration_thresholds': {
                'confidence': 0.30,
                'strength': 0.40
            }
        }
    }

    # Run resumable backtest
    backtest = ResumableBacktest(
        config=config,
        assets=['ETH_4H'],
        start_date='2025-01-01',
        end_date='2025-09-30',
        chunk_days=60
    )

    try:
        results = backtest.run(resume=True)
        print(f"✅ Backtest completed successfully")
        print(f"   Chunks: {results['total_chunks']}")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Return: {results['total_return']:+.2f}%")

    except KeyboardInterrupt:
        print("\n⏸️  Backtest interrupted - progress saved to checkpoint")
        print(f"   Run ID: {backtest.run_id}")
        print(f"   Resume with: python scripts/checkpoint.py resume {backtest.run_id}")

    # Show available checkpoints
    mgr = CheckpointManager()
    checkpoints = mgr.list_checkpoints()

    if checkpoints:
        print(f"\n📁 Available checkpoints:")
        for cp in checkpoints[:5]:  # Show last 5
            print(f"   {cp['run_id']} ({cp['timestamp']})")

if __name__ == "__main__":
    main()