#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:07:13 2018

@author: marcos
"""

import pandas as pd
import csv
import pickle as pkl

# =============================================================================
# Data Manipulation Tool
# =============================================================================
class DMT(object):
      def __init__(self, database_file, file_format='csv', sep=',', decimal='.', orient='index'):
            self.file = database_file
            self.file_format = file_format
            self.sep = sep
            self.decimal = decimal
            self.orient = orient

            if self.file_format == 'csv':
                  self.df = pd.read_csv(self.file, sep=self.sep)
            elif self.file_format == 'json':
                  self.df = pd.read_json(self.file)
            elif self.file_format == 'dict':
                  persisted_dict = pkl.load(open(database_file, 'rb'))
                  self.df = pd.DataFrame.from_dict(persisted_dict, orient=self.orient)

      def print_summary(self):
            print('   Summary of read data:')
            print('-------------------------------------')
            print('%8s | %15s | %8s' % ('Id', 'Name', 'Type'))
            print('-------------------------------------')
            for i,col in enumerate(self.df.dtypes):
                  print('%8d | %15s | %8s' % (i, self.df.columns[i], col))
            print('-------------------------------------')
            print()

      def drop_columns(self, col_list):
            self.data = self.data.drop(columns=col_list)

      def encode_class(self, column):
            self.data[column] = self.data[column].astype(str)

      def save_csv(self, output_file):
            self.data.to_csv(output_file, sep=self.sep, decimal=self.decimal, quoting=csv.QUOTE_NONNUMERIC, index=False)

      def save_json(self, output_file, orient='index'):
            self.data.to_json(output_file, orient=self.orient)

      def save_dict(self, output_file):
            pkl.dump(self.data.to_dict(orient=self.orient), open(output_file, 'wb'))

      def get_json(self):
            return self.data.to_json(orient=self.orient)

      def get_dict(self):
            return self.data.to_dict(orient=self.orient)