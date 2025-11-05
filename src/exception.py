import sys
import logging

# Optional: ensure logging actually writes somewhere
logging.basicConfig(
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)

def error_message_detail(error: Exception, error_detail: sys) -> str:
    # Grab the current traceback from where this is called
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error: {error}"
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    # Use a real f-string (no .format placeholders)
    return (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] | error message: [{error}]"
    )

class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message

# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         # This will log the traceback because exc_info=True
#         logging.error("An error occurred", exc_info=True)
#         # Pass the original exception and sys so we can pull its traceback
#         raise CustomException(e, sys)
