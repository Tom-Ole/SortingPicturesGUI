
from typing import Optional

from PyQt6.QtWidgets import QTextEdit

class Logger:
    """Improved logging with levels and optional GUI output"""
    
    def __init__(self, gui_output: Optional['QTextEdit'] = None):
        self.gui_output = gui_output
        self.enabled = True
    
    def log(self, msg: str, level: str = "INFO"):
        if not self.enabled:
            return
            
        formatted_msg = f"[{level}] {msg}"
        
        if level == "STRONG":
            formatted_msg = f"{'='*50}\n{formatted_msg}\n{'='*50}"
        
        print(formatted_msg)
        
        if self.gui_output:
            self.gui_output.append(formatted_msg)
    
    def info(self, msg: str):
        self.log(msg, "INFO")
    
    def error(self, msg: str):
        self.log(msg, "ERROR")
    
    def strong(self, msg: str):
        self.log(msg, "STRONG")