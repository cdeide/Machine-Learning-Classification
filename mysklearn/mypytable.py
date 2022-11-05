############################################################################
# Name: Connor Deide
# Class: CPSC 322, Fall 2022
# Programming Assignment 5
# 11/2/2022
# Did not attempt the bonus
# 
# Description: This program implements 4 different evaluation approaches for
# evaluating data before forming predictions of unseen instances.
############################################################################

import copy
import csv
from tabulate import tabulate
from mysklearn import myutils

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Get the index of the column to go through
        col_found = False
        for idx in range(len(self.column_names)):
            if self.column_names[idx] == col_identifier:
                col_found = True
                break
        # Raise ValueError if col_identifier was not valid
        if col_found == False:
            raise ValueError("Column was not found")
        
        # Create the column list
        col_list = []
        # Include missing values condition
        if include_missing_values:
            for row in self.data:
                if row[idx] == '':
                    col_list.append("N/A")
                else:
                    col_list.append(row[idx])
        # Exclude missing values condition
        else:
            for row in self.data:
                if row[idx] != '':
                    col_list.append(row[idx])

        return col_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for idx, value in enumerate(row):
                try:
                    row[idx] = float(value)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        if row_indexes_to_drop == []:
            return None
        # Deleting rows in descending order prevents index errors
        row_indexes_to_drop.sort(reverse = True)
        for value in row_indexes_to_drop:
            self.data.pop(value)
            

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # Interact with csv file using csv module
        with open(filename, "r") as infile:
            csv_list = csv.reader(infile)
            for i, row in enumerate(csv_list):
                if i == 0:
                    for value in row:
                        self.column_names.append(value)
                else:
                    self.data.append(row)
        infile.close() # Close file
        self.convert_to_numeric()
        
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w") as outfile:
            write = csv.writer(outfile)

            write.writerow(self.column_names)
            write.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        # Get a list of the indexes to search at
        index_list = []
        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[j]:
                    index_list.append(j)

        # Get a list of the column(s) that we want to work with
        key_columns = []
        for row in self.data:
            new_row = []
            for idx in index_list:
                new_row.append(row[idx])
            key_columns.append(new_row)

        dup_row_idx = []
        for i in range(len(key_columns)):
            j = i + 1
            while(j < len(key_columns)):
                try:
                    if key_columns[i] == key_columns[j]:
                        # Check that index is not in list already
                        Append = True
                        for value in dup_row_idx:
                            if value == j:
                                Append = False
                                break
                        if Append:
                            dup_row_idx.append(j)
                except TypeError:
                    pass
                j += 1

        return dup_row_idx

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        # Iterating rows from bottom to top to omit
        # the problem of the indexes changing as you pop
        row = len(self.data) - 1
        while(row >= 0):
            for value in range(len(self.column_names)):
                if self.data[row][value] == 'NA':
                    self.data.pop(row)
                    break
            row -= 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # Get the index of the column to search through
        col_idx = self.column_names.index(col_name)
        # Find the average of the column
        num = 0
        denom = 0
        for row in self.data:
            if row[col_idx] != 'NA':
                try:
                    num += row[col_idx]
                    denom += 1
                except TypeError:
                    print("Column " + col_name + " does not have continious data")
                    return None
        # Replace missing values with the average
        col_avg = num/denom
        for row in self.data:
            if row[col_idx] == 'NA':
                row[col_idx] = col_avg


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        # Check for empty list
        if self.data == []:
            return self

        # Get a list of the indexes to search at
        index_list = []
        for i in range(len(col_names)):
            for j in range(len(self.column_names)):
                if col_names[i] == self.column_names[j]:
                    index_list.append(j)

        # Get a 2D list of the columns that we want to work with
        summary_stats_list = []
        for i in range(len(index_list)):
            col_list = []
            for row in self.data:
                if row[index_list[i]] != 'NA':
                    col_list.append(row[index_list[i]])
            # Compute stats here
            col_list.sort()
            col_min = min(col_list)
            col_max = max(col_list)
            col_med = (col_min + col_max) / 2
            col_avg = (sum(col_list) / len(col_list))
            # Compute the median 
            # (Code from https://www.geeksforgeeks.org/find-median-of-list-in-python/)
            mid = len(col_list) // 2
            col_median = (col_list[mid] + col_list[~mid]) / 2
            # Put into list
            summary_stats_list.append([col_names[i], col_min, \
                col_max, col_med, col_avg, col_median]) 

        # Create a new MyPyTable with the summary statistics
        summary_stats_table = MyPyTable( \
            ["Attribute", "Max", "Min", "Mid", "Avg", "Median"], summary_stats_list)

        return summary_stats_table

    def get_column_names(self, other_table):
        """Returns a list of column names that is combined from the
            columns names of two tables

        Args:
            other_table (MyPyTable): the second table to join this table with.

        Returns:
            List: a list of the column names for the joined table
        """
        cross_column_names = self.column_names
        for other_name in other_table.column_names:
            duplicate = False
            for name in self.column_names:
                if other_name == name:
                    duplicate = True
                    break
            if not duplicate:
                cross_column_names.append(other_name)
        
        return cross_column_names

    def get_index_lists(self, other_table, key_column_names):
        """Return two lists, each with the indexes of the column names that match the
            column names in key_column_names for each table

        Args:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            List: two lists containg indexes
        """
        index_list_left = []
        index_list_right = []
        for i in range(len(key_column_names)):
            for left_step in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[left_step]:
                    index_list_left.append(left_step)
            for right_step in range(len(other_table.column_names)):
                if key_column_names[i] == other_table.column_names[right_step]:
                    index_list_right.append(right_step)

        return index_list_left, index_list_right

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Get the column_names for the Inner Joins
        cross_column_names = self.get_column_names(other_table)
        # Get a list of the indexes to search at
        index_list_left, index_list_right = self.get_index_lists\
            (other_table, key_column_names)

        inner_join = []
        # Perfrom the Inner Joins
        for row in self.data:
            for other_row in other_table.data:
                match = 0
                for i in range(len(index_list_left)):
                    # Check for a match
                    if row[index_list_left[i]] == other_row[index_list_right[i]]:
                        match += 1
                if match == len(index_list_left):
                    # Match found
                    # Add row to the inner joins table
                    new_row = row.copy()
                    # Don't want to add the key attributes twice
                    for idx in range(len(other_table.column_names)):
                        add_attribute = True
                        for value in index_list_right:
                            if idx == value:
                                add_attribute = False
                        if add_attribute:
                            new_row.append(other_row[idx])
                    inner_join.append(new_row)
            
        inner_join_table = MyPyTable(cross_column_names, inner_join)
        return inner_join_table

    def add_outer_rows(self, other_table, index_list_1, index_list_2, \
        outer_join_table, inner_join_table):
        """Method adds row(s) to the outer_join_table. Checks which rows do
            do not have a match in the corresponding table, fills these rows
            missing values, then inserts them into the outer_join_table

        Args:
            other_table (MyPyTable): the table to compare with
            index_list_1 (List): list of key indexes used for comparisons
            index_list_2 (List): list of key indexes used for comparisons
            inner_join_table (MyPyTable): result of the perform_inner_join method
        """
        for row in self.data:
            add_row = True
            for other_row in other_table.data:
                match = 0
                for i in range(len(index_list_1)):
                    # Check for match
                    if row[index_list_1[i]] == other_row[index_list_2[i]]:
                        match += 1
                if match == len(index_list_1):
                    add_row = False
            if add_row:
                # Add the row
                # Need to fill missing values first
                new_row = []
                for i in range(len(inner_join_table.column_names)):
                    add_missing_val = True
                    for j in range(len(self.column_names)):
                        if inner_join_table.column_names[i] == self.column_names[j]:
                            new_row.append(row[j])
                            add_missing_val = False
                            break
                    if add_missing_val:
                        new_row.append('NA')
                outer_join_table.data.append(new_row)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # Get a list of the indexes to search at
        index_list_left, index_list_right = self.get_index_lists\
            (other_table, key_column_names)
        # First perform an inner joins with a copy
        column_names_copy = self.column_names.copy()
        data_copy = []
        for row in self.data:
            data_copy.append(row)
        table_copy = MyPyTable(column_names_copy, data_copy)
        inner_join_table = table_copy.perform_inner_join(other_table, key_column_names)

        # Perform Full Outer Joins
        outer_join_table = inner_join_table
        self.add_outer_rows(other_table, index_list_left, index_list_right, \
            outer_join_table, inner_join_table)
        other_table.add_outer_rows(self, index_list_right, index_list_left, \
            outer_join_table, inner_join_table)
      
        return outer_join_table