import sys
import logging
from engine import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for backtesting.
    Adjust your strategies by modifying the YAML configs or the logic in engine.py.
    """
    # 1. Initialize Engine
    engine = BacktestEngine()
    
    # 2. Load Configurations (from CLI arguments or default)
    config_paths = sys.argv[1:] if len(sys.argv) > 1 else ['config.yaml']
    engine.load_configs(config_paths)
    
    # 3. Prepare Data (Fetch, Metadata, Clean)
    engine.prepare_data()
    
    # 4. Run Backtests
    engine.run_backtests()
    
    # 5. Plot & Display Results
    engine.plot_all()

if __name__ == "__main__":
    main()
