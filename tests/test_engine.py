from src.engine import BacktestEngine

def test_engine_initialization():
    engine = BacktestEngine()
    assert engine.data_engine is not None
    assert engine.configs == []
