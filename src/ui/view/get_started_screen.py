"""Compatibility entry point: the main screen also handles a missing model."""
from src.ui.view.main_screen import MainScreen

class GetStartedScreen(MainScreen):
    def __init__(self, path=''):
        super().__init__(auto_load=True)
        self.show()
