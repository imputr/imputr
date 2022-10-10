"""
Tests for Table data class.
"""

import pandas as pd
from imputr.domain import Table, DataType

df = pd.read_csv('datasets/unittestsets/DigiDB_digimonlist_small.csv')

def test_ctor_simple():
   table = Table(df)
   
   assert len(table.columns) == 9
   
   for col in table.columns:
       assert col.data.size == 5
       

def test_ctor_with_datatypes():
   table = Table(df, {"Lv 50 Atk": 'cat'})
   
   assert len(table.columns) == 9
   
   for col in table.columns:
       assert col.data.size == 5
       if col.name == 'Lv 50 Atk':
           assert col.type == DataType.CATEGORICAL
    
    