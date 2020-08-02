import xlwt
import xlrd
from datetime import datetime
import requests
import json
import xlutils
from xlutils.copy import copy
#url1="https://api.thingspeak.com/update?api_key=SPYLYPBOT19G7GB9&field1=python"
#result=requests.get(url1)
#result=result.json()
app_id = "384a3378"
app_key = "08577c92e6f727ea81ba655165531e16"
language = "en-gb"
word_id=input("Enter the word : ")
#word_id = "lament"
url = "https://od-api.oxforddictionaries.com:443/api/v2/entries/" + language + "/" + word_id.lower()
r = requests.get(url, headers={"app_id": '384a3378', "app_key": '08577c92e6f727ea81ba655165531e16'})
#print(r)
r=r.json()
r=r['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['definitions'][0]
print(r)
book=xlrd.open_workbook('oxford.xls')
print(book.nsheets)
print(book.sheet_names())
n1=int(input('enter'))
first_sheet = book.sheet_by_index(n1)
len_row=(len(first_sheet.row_values(0)))
len_col=(len(first_sheet.col_values(0)))
print(len_row,len_col)
print(first_sheet.row_values(len_col-1))
print(first_sheet.col_values(len_row-1))

wb=xlwt.Workbook()
ws=wb.add_sheet
ws=wb.add_sheet('sheet2',cell_overwrite_ok=False)
xlutils.copy
ws.write(len_col,len_row,r)
wb.save('oxford.xls')
