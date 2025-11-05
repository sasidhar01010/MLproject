import sys
# Import the configured 'logger' object directly from the 'logger' module
from logger import logger 

# --- Helper Function for Detailed Error Message ---

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including the file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is None:
        return f"Error: {error}"
        
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    return (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] | error message: [{error}]"
    )

# --- Custom Exception Class ---

class CustomException(Exception):
    """
    A custom exception class that logs detailed error messages.
    """
    def __init__(self, error: Exception, error_detail: sys):
        # Pass the exception message up to the base class
        super().__init__(str(error)) 
        
        # Store the formatted, detailed error message
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        """Returns the formatted error message when the exception is printed."""
        return self.error_message

# --- Example Usage ---

if __name__ == "__main__":
    try:
        # Use the imported 'logger' object
        logger.info("Attempting a division operation...") 
        a = 1 / 0  
    except Exception as e:
        # Log the error with full traceback (exc_info=True)
        logger.error("An error occurred during division operation.", exc_info=True)
        
        # Raise the custom exception, which prints the detailed error message
        raise CustomException(e, sys)