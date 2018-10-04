#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:50:33 2018

@author: marcos
"""

import modules.preprocess.dmt as dmt

db_file = 'databases/original/uci/seizure.csv'

db = dmt.DMT(db_file)

print('Antes: %d' % len(db))

db.drop_columns([db.df.columns[0]])
db.set_categorical('y')
db.split_outliers(c=2.0)

print('Depois: %d' % len(db))

db.set_class('y')
db.normalize()

print(db)
