from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


def dt_total(satdd, dataset_name):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)

    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)

    clf = DecisionTreeClassifier()

    x_train = train_data.csr_mat
    y_train = train_data.data_pd['label'].tolist()

    x_test = test_data.csr_mat
    y_test = test_data.data_pd['label'].tolist()

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(dataset_name)
    print(classification_report(y_test, y_pred))