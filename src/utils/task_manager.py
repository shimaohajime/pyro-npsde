from .sheet_io import SheetIO

class TaskManager:
    """
    Provides functionalities for using spreadsheet as a live task list 

    :param sheet: Expects a SheetIO object 
    :param func: Function to call by taking row data as keyword-arguments 

    >>> 
    """
    def __init__(self, sheet, func):
        self.sheet = sheet 
        self.func = func 

