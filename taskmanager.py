import pandas as pd 
from enum import Enum
import ast 

class Task:

    class TaskStatus(Enum):
        UNCLAIMED = "."
        CLAIMED = "/" 
        IN_PROGRESS = "-" 
        COMPLETED = "*" 
        ERROR = "?"
    
    def __init__(self, lock, path, index, status, args):
        self.lock = lock 
        self.path = path 
        self.index = index 
        self.status = status 
        self.args = args 

    def update_status(self, to):
        self.lock.acquire()

        df = pd.read_csv(self.path)
        
        if df.iloc[self.index, 0] != self.status.value:
            raise Exception(f"Task status has been modified by other process. Expected: {self.status.value}. Actual {to.value}")
        
        df.iloc[self.index, 0] = to.value 
        df.to_csv(self.path)
        self.status = to 
        
        self.lock.release() 

def fetch_task_from_tasklist(lock, path):

    # Ensure that only one process accesses the file at a time 
    lock.acquire() 

    try: 
        df = pd.read_csv(path)
        index = query_remaining_sim(df)
        if index == -1:
            return False 

        row_dict = df.iloc[index, 1:].to_dict() 

        for key in row_dict:
            try:
                row_dict[key] = ast.literal_eval(row_dict[key])
            except:
                pass 

        df.iloc[index, 0] = Task.TaskStatus.CLAIMED.value 
        df.to_csv(path)

        return Task(lock, path, index, Task.TaskStatus.CLAIMED, row_dict) 

    finally:
        lock.release() 

        


def query_remaining_sim(df):
    first_col = df.iloc[:,0]
    try:
        return first_col[first_col == "."].index[0]
    except ValueError:
        return -1
