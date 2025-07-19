# src/exception.py

import sys
from src.logger import logger

def error_message_detail(error, error_detail: sys):
    """
    Formats error details to show filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"Error occurred in Python script: {file_name} at line {exc_tb.tb_lineno} -> {str(error)}"
    return error_msg

class CustomException(Exception):
    """
    Custom exception for the pipeline.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
