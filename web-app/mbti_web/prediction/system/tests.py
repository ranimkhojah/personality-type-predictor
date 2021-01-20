import unittest
import pipelines_validator as pipe
import os
import pandas as pd

# Create your tests here.

class TestPipe(unittest.TestCase):
    
#    def setup(self):

    def test_createValidationSchema_csvDoesntExist(self):
        file= 'test_files/a_file_that_doesnt_exist.csv'
        self.assertRaises(FileNotFoundError, pipe.createValidationSchema(file))
 
    def test_createValidationSchema_schemaCreated(self):
        file= 'test_files/test_data.csv'
        json_file = 'schema.json'
        if os.path.exists(json_file):
            os.remove(json_file)
        if os.path.exists('schema-inferred.json'):
            os.remove('schema-inferred.json')
        result = pipe.createValidationSchema(file)
        self.assertEqual(result, 1) 
 
    def test_schema_validateServingData_fileDoesntExist(self):
        sch= 'schema.json'
        file = 'test_files/file_that_doesnt_exist.csv'
        self.assertRaises(FileNotFoundError, pipe.schema_validateServingData(sch, file))

    def test_schema_validateServingData_SchemaDoesntExist(self):
        sch= 'test_files/schema_that_doesnt_exist.json'
        file = 'test_files/test_data.csv'
        self.assertRaises(FileExistsError, pipe.schema_validateServingData(sch, file))

    def test_schema_validateServingData_ErrorsExist(self):
        sch= 'test_files/mock_schema.json'
        file = 'test_files/data_with_wrong_types.csv'
        numError = pipe.schema_validateServingData(sch, file)
        #expected three errors: string replaces a boolean/ int replaces string/ string replaces int
        self.assertEqual(10, numError)
   
    def test_valuesValidation_duplicates_and_types(self):
        df = pd.read_csv('test_files/dup_mistype.csv', encoding="utf-8")
        result = pipe.valuesValidation(df)
        self.assertEqual(result, 4)

    def test_valuesValidation_dupliacte_posts(self):
        df = pd.read_csv('test_files/just_dup.csv', encoding="utf-8")
        result = pipe.valuesValidation(df)
        self.assertEqual(result, 1)

    def test_valuesValidation_wrong_type(self):
        df = pd.read_csv('test_files/just_mistype.csv', encoding="utf-8")
        result = pipe.valuesValidation(df)
        self.assertEqual(result, 2)

    def test_valuesValidation_null_values(self):
        df = pd.read_csv('test_files/nulls.csv', encoding="utf-8")
        result = pipe.valuesValidation(df)
        self.assertEqual(result, 3)

unittest.main(argv=[''], exit=False)

