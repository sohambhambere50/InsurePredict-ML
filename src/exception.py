import sys
from types import TracebackType
from typing import Optional


def error_message_detail(error: Exception, error_detail):
    """
    Returns detailed error message with file name and line number
    """
    exc_type, exc_value, exc_tb = sys.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown file"
        line_number = "Unknown line"

    return (
        f"Error occurred in script: [{file_name}] "
        f"at line [{line_number}] "
        f"error message: [{str(error)}]"
    )


class CustomException(Exception):
    """
    Custom exception class for better error tracking
    """

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, sys)

    def __str__(self):
        return self.error_message
