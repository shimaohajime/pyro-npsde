# import pandas as pd 

def validate_input_dataframe(df):
    assert df.columns[0] == 'Label' and df.columns[1] == 'Time', "Input data has invalid format"

def validate_tasklist(df):
    assert df.columns[0] == 'Status', "Tasklist has invalid format"

def append_to_report(state, lines):
    if "report" not in state:
        return 

    state["report_lock"].acquire() 

    with open(state["report"], "a") as f:
        for line in lines:
            f.write(line + "\n")

    f.close() 
    state["report_lock"].release() 
            
