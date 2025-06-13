import os
import logging
from datetime import datetime
from typing import List, Dict, Any

class SignalEngineLogger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs_and_tests"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up logging configuration
        self.logger = logging.getLogger('signal_engine')
        self.logger.setLevel(logging.INFO)
        
        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"signal_engine_{timestamp}.log")
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def log_signal_engine_run(self, 
                            assets_processed: List[str],
                            signal_count: int,
                            macro_bias_score: float,
                            errors: List[str] = None):
        """
        Log the results of a signal engine run
        
        Args:
            assets_processed: List of asset symbols that were processed
            signal_count: Number of signals generated
            macro_bias_score: Current macro bias score
            errors: List of any errors that occurred during processing
        """
        self.logger.info("=== Signal Engine Run Summary ===")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Assets Processed: {', '.join(assets_processed)}")
        self.logger.info(f"Total Signals Generated: {signal_count}")
        self.logger.info(f"Macro Bias Score: {macro_bias_score}")
        
        if errors:
            self.logger.error("Errors encountered:")
            for error in errors:
                self.logger.error(f"- {error}")
        
        self.logger.info("=== End of Run Summary ===\n")
    
    def log_error(self, error_message: str):
        """Log an error message"""
        self.logger.error(error_message)
    
    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message) 