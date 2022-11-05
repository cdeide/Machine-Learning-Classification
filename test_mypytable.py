"""test_mypytable.py

@author gsprint23
Note: do not modify this file
"""

import os
import copy
import pandas as pd

from mysklearn.mypytable import MyPyTable

# test input data
header = ["id", "a", "b", "c"]
data = [["ID001", "1", "-5.5", 1.7],
            ["ID002", "A", "2.2", 1.0]]
data_as_numeric = [["ID001", 1, -5.5, 1.7],
            ["ID002", "A", 2.2, 1.0]]

header_dups = ["id", "a", "b"]
data_dups = [["ID001", "A", 1.0],
            ["ID002", "B", 1.5],
            ["ID003", "A", 1.0],
            ["ID002", "A", 1.5],
            ["ID004", "C", 1.0],
            ["ID005", "C", 1.0],
            ["ID006", "D", 1.0],
            ["ID007", "A", 2.0],
            ["ID008", "C", 1.0]]

data_dups_dropped = [["ID001", "A", 1.0],
            ["ID002", "B", 1.5],
            ["ID002", "A", 1.5],
            ["ID004", "C", 1.0],
            ["ID006", "D", 1.0],
            ["ID007", "A", 2.0]]

header_stats = ["a", "b", "c"]
data_stats = [[1.0, 2.0, 3.0],
                [2.5, 2.0, 1.0],
                [0.0, -1.0, 1.0],
                [-2.0, 0.5, 0.0]]

# first trace example (single key)
# adapted from SQL examples at https://www.diffen.com/difference/Inner_Join_vs_Outer_Join
header_left = ["Product", "Price"]
data_left = [["Potatoes", 3.0],
                ["Avacodos", 4.0],
                ["Kiwis", 2.0],
                ["Onions", 1.0],
                ["Melons", 5.0],
                ["Oranges", 5.0],
                ["Tomatoes", 6.0]]
header_right = ["Product", "Quantity"]
data_right = [["Potatoes", 45.0],
                ["Avacodos", 63.0],
                ["Kiwis", 19.0],
                ["Onions", 20.0],
                ["Melons", 66.0],
                ["Broccoli", 27.0],
                ["Squash", 92.0]]

# second trace example (multiple attribute key)
header_car_left = ["SaleId","EmployeeId","CarName","ModelYear","Amt"]
data_car_left = [[555.0,12.0,"ford pinto",75.0,3076.0],
                [556.0,12.0,"toyota corolla",75.0,2611.0],
                [998.0,13.0,"toyota corolla",75.0,2800.0],
                [999.0,12.0,"toyota corolla",76.0,2989.0]]
header_car_right = ["CarName","ModelYear","MSRP"]
data_car_right = [["ford pinto",75.0,2769.0],
                    ["toyota corolla",75.0,2711.0],
                    ["ford pinto",76.0,3025.0],
                    ["toyota corolla",77.0,2789.0]]

# join practice problem
header_car_left_long = ["SaleId","EmployeeId","CarName","ModelYear","Amt"]
data_car_left_long = [[555.0,12.0,"ford pinto",75.0,3076.0], # full match
    [556.0,12.0,"toyota truck",79.0,2989.0], # 0 match
    [557.0,12.0,"toyota corolla",75.0,2611.0], # full match
    [996.0,13.0,"toyota corolla",75.0,2800.0], # 2nd full match
    [997.0,12.0,"toyota corolla",76.0,2989.0], # match on CarName, match on ModelYear; not together
    [998.0,12.0,"ford pinto",74.0,2989.0], # match on CarName, 0 match on ModelYear
    [999.0,12.0,"ford mustang",77.0,2989.0]] # 0 match on CarName, match on ModelYear
header_car_right_long = ["CarName","ModelYear","MSRP"]
data_car_right_long = [["honda accord",75.0,2789.0], # 0 match on CarName, match on ModelYear
    ["ford pinto",75.0,2769.0], # full match
    ["toyota corolla",75.0,2711.0], # full match
    ["ford pinto",76.0,3025.0], # match on CarName, match on ModelYear; not together
    ["toyota corolla",77.0,2789.0], # match on CarName, match on ModelYear; not together
    ["range rover",70.0,3333.0], # 0 match
    ["ford pinto",73.0,2567.0], # match on CarName, 0 match on ModelYear
    ["toyota corolla",75.0,2999.0]] # 2nd full match

header_NAs = ["id", "a", "b"]
data_NAs = [["ID001", "A", 3.5],
            ["ID002", "B", "NA"],
            ["ID003", "C", 1.0],
            ["ID004", "D", 1.5]]

header_NAs2 = ["id", "a", "b", "c"]
data_NAs2 = [["ID001", 3.5, 5, 1.1],
            ["ID002", 2.2, 7, -5.3],
            ["ID003", 1.0, "NA", 20.20],
            ["ID004", "NA", -5, 100.101],
            ["ID005", 1.4, 5, "NA"],
            ["ID006", "NA", 50, 0.5]]

# note: order is actual/received student value, expected/solution
# def test_table_init():
# """Test function
# """
#     table = MyPyTable(header, data)

#     assert len(table.column_names) == len(header)
#     assert len(table.data) == 2

#     # make sure deep copies are made
#     # id() returns unique integer for objects in memory
#     # https://docs.python.org/3/library/functions.html#id
#     assert id(table.data) != id(data)
#     for i in range(len(table.data)):
#         assert id(table.data[i]) != id(data[i])

#     # test empty
#     table = MyPyTable([], [])
#     assert len(table.column_names) == 0
#     assert len(table.data) == 0

def test_get_shape():
    """Test function
    """
    table = MyPyTable(header, data_as_numeric)

    shape = table.get_shape()
    assert shape[0] == len(data)
    assert shape[1] == len(header)

    # test empty
    table = MyPyTable([], [])
    assert len(table.column_names) == 0
    assert len(table.data) == 0

def test_get_column():
    """Test function
    """
    table = MyPyTable(header, data_as_numeric)

    for i, col_label in enumerate(header):
        col = table.get_column(col_label)
        assert len(col) == len(data_as_numeric)

        for j, _ in enumerate(col):
            assert col[j] == data_as_numeric[j][i]

    # test empty
    table = MyPyTable(header, [])
    col = table.get_column("a")
    assert len(col) == 0

def test_convert_to_numeric():
    """Test function
    """
    table = MyPyTable(header, data)
    table.convert_to_numeric()

    assert len(table.data) == len(data_as_numeric)
    assert table.data == data_as_numeric

def test_drop_rows():
    """Test function
    """
    table = MyPyTable(header_dups, data_dups)
    rows_to_drop = [3]
    table.drop_rows(rows_to_drop)
    temp = data_dups.copy() # shallow copy
    temp.pop(3)
    assert table.data == temp

    table = MyPyTable(header_dups, data_dups)
    rows_to_drop = [2, 5, 8]
    table.drop_rows(rows_to_drop)
    assert table.data == data_dups_dropped

    # test empty
    table = MyPyTable(header_dups, data_dups)
    table.drop_rows([])
    assert table.data == data_dups

def test_load_from_file():
    """Test function
    """
    fname = os.path.join("test_data", "dummy.csv")
    table = MyPyTable().load_from_file(fname)

    assert len(table.column_names) == len(header)
    assert len(table.data) == len(data_as_numeric)
    assert table.data == data_as_numeric

def test_save_to_file():
    """Test function
    """
    fname = os.path.join("test_data", "dummy_out.csv")
    table = MyPyTable(header, data_as_numeric)

    table.save_to_file(fname)
    with open(fname, "r") as infile:
        lines = infile.readlines()

        assert len(lines) == len(data_as_numeric) + 1 # + 1 for header
        assert lines == ['id,a,b,c\n', 'ID001,1,-5.5,1.7\n', 'ID002,A,2.2,1.0\n']

def test_find_duplicates():
    """Test function
    """
    table = MyPyTable(header_dups, data_dups)

    row_indexes_to_drop = table.find_duplicates(["id"])
    assert row_indexes_to_drop == [3]

    row_indexes_to_drop = table.find_duplicates(["a", "b"])
    assert row_indexes_to_drop == [2, 5, 8]

    # test empty
    table = MyPyTable(header_dups, [])
    row_indexes_to_drop = table.find_duplicates(["a"])
    assert row_indexes_to_drop == []

def test_remove_rows_with_missing_values():
    """Test function
    """
    # one row with missing value
    table = MyPyTable(header_NAs, data_NAs)
    table.remove_rows_with_missing_values()
    assert len(table.data) == 3
    removed_nas = [["ID001", "A", 3.5],
            ["ID003", "C", 1.0],
            ["ID004", "D", 1.5]]
    assert table.data == removed_nas

    # several rows of missing values
    table = MyPyTable(header_NAs2, data_NAs2)
    table.remove_rows_with_missing_values()
    assert len(table.data) == 2
    removed_nas2 = [["ID001", 3.5, 5, 1.1],
            ["ID002", 2.2, 7, -5.3]]
    assert table.data == removed_nas2

    # test on table w/o missing vals
    table = MyPyTable(header, data)
    table.remove_rows_with_missing_values()
    assert len(table.data) == 2
    assert table.data == data

def test_replace_missing_values_with_column_average():
    """Test function
    """
    table = MyPyTable(header_NAs, data_NAs)

    table.replace_missing_values_with_column_average("b")
    assert len(table.data) == 4

    replaced_nas = [["ID001", "A", 3.5],
            ["ID002", "B", 2.0],
            ["ID003", "C", 1.0],
            ["ID004", "D", 1.5]]
    assert table.data == replaced_nas

def test_compute_summary_statistics():
    """Test function
    """
    table = MyPyTable(header_stats, data_stats)

    # even number of instances
    stats_table = table.compute_summary_statistics(["a", "c"])
    assert stats_table.data == [['a', -2.0, 2.5, 0.25, 0.375, 0.5], ['c', 0.0, 3.0, 1.5, 1.25, 1.0]]

    # odd number of instances
    data_stats_copy = data_stats.copy()
    data_stats_copy.pop(0)
    table = MyPyTable(header_stats, data_stats_copy)

    stats_table = table.compute_summary_statistics(["b"])
    assert stats_table.data == [['b', -1.0, 2.0, 0.5, 0.5, 0.5]]

    # test empty
    table = MyPyTable(header, [])
    stats_table = table.compute_summary_statistics(["b"])
    assert stats_table.data == []

def check_same_lists_regardless_of_order(list1, list2):
    """Utility function
    """
    assert len(list1) == len(list2) # same length
    for item in list1:
        assert item in list2
        list2.remove(item)
    assert len(list2) == 0
    return True

def test_perform_inner_join():
    """Test function
    """
    # single attribute key
    table_left = MyPyTable(header_left, data_left)
    table_right = MyPyTable(header_right, data_right)
    joined_table = table_left.perform_inner_join(table_right, ["Product"])
    assert len(joined_table.column_names) == 3
    # test against pandas' inner join
    df_left = pd.DataFrame(data_left, columns=header_left)
    df_right = pd.DataFrame(data_right, columns=header_right)
    df_joined = df_left.merge(df_right, how="inner", on=["Product"])
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # multiple attribute key example from class
    table_left = MyPyTable(header_car_left, data_car_left)
    table_right = MyPyTable(header_car_right, data_car_right)
    joined_table = table_left.perform_inner_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' inner join
    df_left = pd.DataFrame(data_car_left, columns=header_car_left)
    df_right = pd.DataFrame(data_car_right, columns=header_car_right)
    df_joined = df_left.merge(df_right, how="inner", on=["CarName", "ModelYear"])
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # join practice problem
    table_left = MyPyTable(header_car_left_long, data_car_left_long)
    table_right = MyPyTable(header_car_right_long, data_car_right_long)
    joined_table = table_left.perform_inner_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' inner join
    df_left = pd.DataFrame(data_car_left_long, columns=header_car_left_long)
    df_right = pd.DataFrame(data_car_right_long, columns=header_car_right_long)
    df_joined = df_left.merge(df_right, how="inner", on=["CarName", "ModelYear"])
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # now check non-adjacent composite key columns
    # data prep
    header_car_left_copy = copy.deepcopy(header_car_left_long)
    data_car_left_copy = copy.deepcopy(data_car_left_long)
    header_car_right_copy = copy.deepcopy(header_car_right_long)
    data_car_right_copy = copy.deepcopy(data_car_right_long)
    # swap CarName and SaleId columns,
    # then CarName and ModelYear columns so they are in diff order, 2 apart
    header_car_left_copy[0], header_car_left_copy[2] = \
        header_car_left_copy[2], header_car_left_copy[0]
    header_car_left_copy[0], header_car_left_copy[3] = \
        header_car_left_copy[3], header_car_left_copy[0]
    for row in data_car_left_copy:
        row[0], row[2] = row[2], row[0]
        row[0], row[3] = row[3], row[0]
    # swap ModelYear and MSRP columns so they are 1 apart
    header_car_right_copy[1], header_car_right_copy[2] = \
        header_car_right_copy[2], header_car_right_copy[1]
    for row in data_car_right_copy:
        row[1], row[2] = row[2], row[1]
    # test setup
    table_left = MyPyTable(header_car_left_copy, data_car_left_copy)
    table_right = MyPyTable(header_car_right_copy, data_car_right_copy)
    joined_table = table_left.perform_inner_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' inner join
    df_left = pd.DataFrame(data_car_left_copy, columns=header_car_left_copy)
    df_right = pd.DataFrame(data_car_right_copy, columns=header_car_right_copy)
    df_joined = df_left.merge(df_right, how="inner", on=["CarName", "ModelYear"])
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

def test_perform_full_outer_join():
    """Test function
    """
    # single attribute key
    table_left = MyPyTable(header_left, data_left)
    table_right = MyPyTable(header_right, data_right)
    joined_table = table_left.perform_full_outer_join(table_right, ["Product"])
    assert len(joined_table.column_names) == 3
    # test against pandas' outer join
    df_left = pd.DataFrame(data_left, columns=header_left)
    df_right = pd.DataFrame(data_right, columns=header_right)
    df_joined = df_left.merge(df_right, how="outer", on=["Product"])
    df_joined.fillna("NA", inplace=True)
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # multiple attribute key example from class
    table_left = MyPyTable(header_car_left, data_car_left)
    table_right = MyPyTable(header_car_right, data_car_right)
    joined_table = table_left.perform_full_outer_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' inner join
    df_left = pd.DataFrame(data_car_left, columns=header_car_left)
    df_right = pd.DataFrame(data_car_right, columns=header_car_right)
    df_joined = df_left.merge(df_right, how="outer", on=["CarName", "ModelYear"])
    df_joined.fillna("NA", inplace=True)
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # join practice problem
    table_left = MyPyTable(header_car_left_long, data_car_left_long)
    table_right = MyPyTable(header_car_right_long, data_car_right_long)
    joined_table = table_left.perform_full_outer_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' outer join
    df_left = pd.DataFrame(data_car_left_long, columns=header_car_left_long)
    df_right = pd.DataFrame(data_car_right_long, columns=header_car_right_long)
    df_joined = df_left.merge(df_right, how="outer", on=["CarName", "ModelYear"])
    df_joined.fillna("NA", inplace=True)
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # now check non-adjacent composite key columns
    # data prep
    header_car_left_copy = copy.deepcopy(header_car_left_long)
    data_car_left_copy = copy.deepcopy(data_car_left_long)
    header_car_right_copy = copy.deepcopy(header_car_right_long)
    data_car_right_copy = copy.deepcopy(data_car_right_long)
    # swap CarName and SaleId columns, then CarName and ModelYear columns
    # so they are in diff order, 2 apart
    header_car_left_copy[0], header_car_left_copy[2] = \
        header_car_left_copy[2], header_car_left_copy[0]
    header_car_left_copy[0], header_car_left_copy[3] = \
        header_car_left_copy[3], header_car_left_copy[0]
    for row in data_car_left_copy:
        row[0], row[2] = row[2], row[0]
        row[0], row[3] = row[3], row[0]
    # swap ModelYear and MSRP columns so they are 1 apart
    header_car_right_copy[1], header_car_right_copy[2] = \
        header_car_right_copy[2], header_car_right_copy[1]
    for row in data_car_right_copy:
        row[1], row[2] = row[2], row[1]
    # test setup
    table_left = MyPyTable(header_car_left_copy, data_car_left_copy)
    table_right = MyPyTable(header_car_right_copy, data_car_right_copy)
    joined_table = table_left.perform_full_outer_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' outer join
    df_left = pd.DataFrame(data_car_left_copy, columns=header_car_left_copy)
    df_right = pd.DataFrame(data_car_right_copy, columns=header_car_right_copy)
    df_joined = df_left.merge(df_right, how="outer", on=["CarName", "ModelYear"])
    df_joined.fillna("NA", inplace=True)
    check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())
