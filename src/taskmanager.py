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

    def update(self, to=None, columns=None):
        self.lock.acquire()

        df = pd.read_csv(self.path)

        if df.iloc[self.index, 0] != self.status.value:
            raise Exception(f"Task status has been modified by other process. Expected: {self.status.value}. Actual {to.value}")

        if to:
            df.iloc[self.index, 0] = to.value

        if columns:
            for key, value in columns:
                df.iloc[self.index, :].loc[key] = value

        df.to_csv(self.path, index=False)
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
        df.to_csv(path, index=False)

        return Task(lock, path, index, Task.TaskStatus.CLAIMED, row_dict)

    finally:
        pass
        lock.release()




def query_remaining_sim(df):
    first_col = df.iloc[:,0]
    try:
        return first_col[first_col == "."].index[0]
    except:
        return -1
