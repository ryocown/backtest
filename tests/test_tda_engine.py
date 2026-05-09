from src.tda_engine import TDAManager

def test_tda_manager_init():
    manager = TDAManager(window_months=6)
    assert manager.window_months == 6
