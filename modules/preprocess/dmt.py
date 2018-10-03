#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:07:13 2018

@author: marcos
"""

import pandas as pd
import csv
import pickle as pkl
import numpy as np
import scipy.stats as sts
from sklearn import preprocessing as prep

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
            self.classes = None
            self.minima = None
            self.maxima = None

            if self.file_format == 'csv':
                  self.df = pd.read_csv(self.file, sep=self.sep)
            elif self.file_format == 'json':
                  self.df = pd.read_json(self.file)
            elif self.file_format == 'dict':
                  persisted_dict = pkl.load(open(database_file, 'rb'))
                  self.df = pd.DataFrame.from_dict(persisted_dict, orient=self.orient)

      ############ I/O and Import/Export Methods ####################
      def print_summary(self):
            print('   Summary of read data:')
            print('-------------------------------------')
            print('%8s | %15s | %8s' % ('Id', 'Name', 'Type'))
            print('-------------------------------------')
            for i,col in enumerate(self.df.dtypes):
                  print('%8d | %15s | %8s' % (i, self.df.columns[i], col))
            print('-------------------------------------')
            print()

      def save_csv(self, output_file):
            self.df.to_csv(output_file, sep=self.sep, decimal=self.decimal, quoting=csv.QUOTE_NONNUMERIC, index=False)

      def save_json(self, output_file, orient='index'):
            self.df.to_json(output_file, orient=self.orient)

      def save_dict(self, output_file):
            pkl.dump(self.df.to_dict(orient=self.orient), open(output_file, 'wb'))

      def get_json(self):
            return self.df.to_json(orient=self.orient)

      def get_dict(self):
            return self.df.to_dict(orient=self.orient)

      ############ Column manipulation Methods ####################
      def drop_columns(self, col_list):
            self.df = self.df.drop(columns=col_list)

      def encode_class(self, column):
            self.df[column] = self.df[column].astype(str)
            self.classes = self.df[column].copy()
            self.df.drop(columns=[column])

      def is_classes_set(self):
            return self.classes is not None

      def get_classes(self):
            return self.classes

      # Encode categorical data into integer ids
      def encode_categorical(self):
          le = prep.LabelEncoder()
          for x in self.df.columns:
              if self.df[x].dtypes == 'object':
                  self.df[x] = le.fit_transform(self.df[x])

      ############ Data Transformation Methods ####################
      def normalize(self):
            numeric_data = self.df._get_numeric_data()
            maxima = numeric_data.max()
            minima = numeric_data.min()

            data_range = maxima - minima
            data_range[data_range == 0] = 1.0

            numeric_data = (numeric_data - minima) / data_range

            self.df[numeric_data.columns] = numeric_data

            self.minima = minima
            self.maxima = maxima

      def denormalize(self):
            if (self.minima is not None) and (self.maxima is not None):
                  numeric_data = self.df._get_numeric_data()
                  numeric_data = numeric_data * (self.maxima - self.minima) + self.minima
                  self.df[numeric_data.columns] = numeric_data

      def split_outliers(self, limQ1=25, limQ3=75, c=1.5):
            numeric_data = self.df._get_numeric_data()

            q1 = np.percentile(numeric_data, limQ1, axis=0)
            q3 = np.percentile(numeric_data, limQ3, axis=0)
            iqr = sts.iqr(numeric_data, axis=0)

            # Até aqui OK

            new_data = []
            ix_data = []
            outliers_inf = []
            ix_inf = []
            outliers_sup = []
            ix_sup = []

            i = 0
            for d in data:
                  test_inf = d < q1 - c * iqr
                  if test_inf.any():
                        outliers_inf.append(d)
                        ix_inf.append(i)
                  else:
                        test_sup = d > q3 + c * iqr
                        if test_sup.any():
                              outliers_sup.append(d)
                              ix_sup.append(i)
                        else:
                              new_data.append(d)
                              ix_data.append(i)
                  i += 1

            new_data = pd.DataFrame(np.array(new_data), index=numeric_data.index[ix_data], columns=numeric_data.columns)
            outliers_inf = pd.DataFrame(np.array(outliers_inf), index=numeric_data.index[ix_inf], columns=numeric_data.columns)
            outliers_sup = pd.DataFrame(np.array(outliers_sup), index=numeric_data.index[ix_sup], columns=numeric_data.columns)

            return new_data, outliers_inf, outliers_sup
