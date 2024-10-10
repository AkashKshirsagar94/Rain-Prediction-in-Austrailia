import sys

def error_message_detail(error):
    """
    Generates a detailed error message, including file name, line number, and error message.

    Args:
        error: The error object.
        error_detail: The exception information tuple (optional).

    Returns:
        A formatted error message string.
    """

    _, _, exc_tb = sys.exc_info()
    error_detail = (None, None, exc_tb)

    file_name = error_detail[2].tb_frame.f_code.co_filename
    line_number = error_detail[2].tb_lineno
    error_message = f"Error occurred in Python script '{file_name}' on line {line_number}: {str(error)}"

    return error_message