import pandas as pd 
from enum import Enum

class Task:

    class TaskStatus(Enum):
        UNCLAIMED = ""
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
        
        if df.iloc[self.index, 0] != self.status.name:
            raise Exception(f"Task status has been modified by other process. Expected: {self.status.name}. Actual {to.name}")
        
        df.iloc[self.index, 0] = to 
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

        return Task(lock, path, index, Task.TaskStatus.CLAIMED, row_dict) 

    finally:
        lock.release() 

        


def query_remaining_sim(df):
    first_col = df.iloc[:,0]
    try:
        return first_col.index("") 
    except ValueError:
        return -1
