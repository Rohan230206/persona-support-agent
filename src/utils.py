import logging
import sys
import time
import random
import functools
from typing import Callable, Any, TypeVar
from pypdf import PdfReader
from src import config

# Initialize generic TypeVar for typing decorator
F = TypeVar('F', bound=Callable[..., Any])

# Configure Logging
def setup_logger(name: str = "support_agent") -> logging.Logger:
    """Sets up console and file logging."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        try:
            file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file log handler: {e}")
            
    return logger

# Get application logger
logger = setup_logger()

def retry_with_backoff(max_retries: int = 5, initial_delay: float = 1.0, backoff_factor: float = 2.0) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff on failure.
    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Delay before first retry in seconds.
        backoff_factor (float): Multiplier for delay calculation on subsequent retries.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for function '{func.__name__}': {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    # Jitter sleep to avoid simultaneous retries
                    time.sleep(delay + random.uniform(0.0, 0.5))
                    delay *= backoff_factor
            
            logger.error(f"All {max_retries} attempts failed for function '{func.__name__}'.")
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content page-by-page from a PDF document.
    Args:
        pdf_path (str): Absolute or relative path to the PDF file.
    Returns:
        str: Extracted text contents.
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        extracted_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_text.append(text)
            else:
                logger.warning(f"No text extracted from page {i} of {pdf_path}")
        
        full_text = "\n".join(extracted_text)
        logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path}")
        return full_text
    except Exception as e:
        logger.error(f"Failed to read PDF file {pdf_path}: {e}")
        raise RuntimeError(f"Error parsing PDF file: {e}")
