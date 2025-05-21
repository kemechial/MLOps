from src.logger import get_logger
from src.custom_exception import CustomException

import sys

logger = get_logger(__name__)

def divide(x, y):
    try:
        result = x / y
        logger.info(f"Division successful: {x} / {y} = {result}")
        return result
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise CustomException("An unexpected error occurred", sys) 
    
if __name__ == "__main__":
    try:
        logger.info("Starting the division operation")
        divide(10, 0)
    except CustomException as ce:
        logger.error(str(ce))