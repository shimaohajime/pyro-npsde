import os 
import gspread 
import pandas as pd 
from oauth2client.service_account import ServiceAccountCredentials 

class SheetIO:
    """
        Flexible abstraction of spreadsheet IO, allows interaction with both local excel files and Google spreadsheet

        >>> mySheet = sheetIO(path="localsheet.xls")
        >>> mySheet = sheetIO(keyfile="serviceacc.json", gspread_name="gspread_sheet")
        
        :param sheet_name: Name of sheet in workbook 
        :param path: Path to local excel file
        :param keyfile: Path to Google service account keyfile 
        :param gspread_name: Path to Google workbook name 

        :raises FileNotFoundError: if provided path does not exist 
        :raises ValueError: if no valid set of arguments is given 
        
    """

    def __init__(self, sheet_name, path=None, keyfile=None, gspread_name=None):

        if path is not None:
            # Local spreadsheet 
            self.type = "local"
            if not os.path.exists(path):
                raise FileNotFoundError()
            self.path = path 
            self.sheet_name = sheet_name 
        elif keyfile is not None and gspread_name is not None:
            # Google sheet 
            self.type = "gspread"
            self.sheet = _get_sheets(keyfile, gspread_name).worksheet(sheet_name)
        else:
            raise ValueError("Must provide either path to local spreadsheet or keyfile + Google spreadsheet name")


    def _get_sheets(secret_json_file, file_name):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(secret_json_file, scope)
        client = gspread.authorize(creds)
        sheets = client.open(file_name)
        return sheets

    def __setitem__(self, key, value):
        if self.type == "local":
            df = pd.read_excel(self.path, self.sheet_name)
            df[key] = value 
            df.to_excel(self.path, self.sheet_name)
        elif self.type == "gspread":
            pass 

def query_remaining_sim(sheet):
    
    first_col = sheet.col_values(1)
    print(first_col)
    try:
        return first_col.index("") + 1
    except ValueError:
        if len(first_col) < len(list(filter(None, sheet.col_values(2)))):
            return len(first_col) + 1 
        else:
            return -1


def execute_next_task(sheet_tups, task, **kwargs):
    sheet = get_sheet(*sheet_tups)
    sim_index = query_remaining_sim(sheet)
    if sim_index == -1:
        return False 
    
    headers = sheet.row_values(1)
    row_dict = {}
    row_val = sheet.row_values(sim_index)
    for i in range(len(headers)):
        if i >= len(row_val):
            row_dict[headers[i]] = ''
            continue
        try:
            row_dict[headers[i]] = ast.literal_eval(str(row_val[i]))
        except:
            row_dict[headers[i]] = str(row_val[i])
    task(sheet_tups, sim_index, row_dict, **kwargs)
    return True 

    def populate_instructions(csv_file, runs, sheet_tup):
        sheet = get_sheet(*sheet_tup)
        data = pd.read_csv(csv_file)
        headers = [sheet.cell(1,i).value for i in range(1,sheet.col_count+1)]
        corr_col = [headers.index(h) for h in data.columns]
        # start_row  = len(list(filter(None, sheet.col_values(1)))) + 1 
        # Ensure that sheet has all the required headers 
        assert(not(any([x == -1 for x in corr_col])))

        for _ in range(runs):
            for r in range(data.shape[0]):
                added_row = [''] * sheet.col_count
                for i in range(len(corr_col)):
                    added_row[corr_col[i]] = data.iloc[r,i]
                sheet.append_row([str(x) for x in added_row]) 