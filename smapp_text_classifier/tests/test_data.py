from smapp_text_classifier.data import DataSet

def data_set_tests(dataset):
    assert dataset.df_train.shape[0] == 4
    assert dataset.df_test.shape[0] == 2

def test_data_set_from_json():
    dataset = DataSet(input_='test_data.json')
    data_set_tests(dataset)

def test_data_set_from_csv():
    dataset = DataSet(input_='test_data.csv')
    data_set_tests(dataset)

def test_data_set_from_tsv():
    dataset = DataSet(input_='test_data.tsv')
    data_set_tests(dataset)

