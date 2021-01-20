import pandas as pd
import sqlite3
from sqlite3 import Error

# Author: Salvatore
# Establish and returns connection to db
def db_connection(db_name):
    conn=None
    try:
        conn= sqlite3.connect(db_name)
        return conn
    except Error as e:
        print(e)
    return conn

#Author: Salvatore
def db_close(db_connection):
    try:
        db_connection.close()
        print("Database connection successfully closed.")
    except Error as e:
        print("Error while closing the database connection")
        print(e)

# Author: Salvatore
def create_user_table(cursor):
    try:
        cursor.execute('''
                CREATE TABLE user(
                     id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                     name VARCHAR (20) NOT NULL,
                     user_type INT NOT NULL,
                     password VARCHAR(30) NOT NULL,
                     email VARCHAR (50)
                    )''')
    except Error as e:
        print("Error while creating user table")
        print(e)


# Author: Salvatore
def create_prediction_table(cursor):
    try:
        cursor.execute('''
                CREATE TABLE prediction(
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    type_predicted INT NOT NULL,
                    user_id INT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user (id)
                     )''')
    except Error as e:
        print("Error while creating prediction table")
        print(e)

# Author: Salvatore
def create_prediction_model_table(cursor):
    try:
        cursor.execute('''
                CREATE TABLE prediction_model(
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    accuracy real NOT NULL
                    )''')
    except Error as e:
        print("Error while creating prediction model table")
        print(e)

# Author: Salvatore
def create_package_table(cursor):
    try:
        cursor.execute('''
                CREATE TABLE package(
                    name text NOT NULL PRIMARY KEY,
                    version real NOT NULL,
                    prediction_model_id INT,
                    FOREIGN KEY (prediction_model_id) REFERENCES prediction_model (id)
                    )''')
    except Error as e:
        print("Error while creating packages table")
        print(e)

# Author: Salvatore
def create_training_data_table(cursor):
    try:
        cursor.execute('''
                CREATE TABLE training_data(
                    training_row INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    type INT NOT NULL,
                    clean_post  text NOT NULL,
                    prediction_model_id INT,
                    dirty_row INT NOT NULL,
                    
                    FOREIGN KEY (prediction_model_id) REFERENCES prediction_model (id)
                    )''')
    except Error as e:
        print("Error while creating user table")
        print(e)

# Author: Salvatore
def create_dirty_data_table(cursor):
    try:
        cursor.execute('''
                CREATE TABLE dirty_data(
                    dirty_row INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    type VARCHAR(4) NOT NULL,
                    dirty_post  text NOT NULL,
                    clean_row INT
                    )''')
    except Error as e:
        print("Error while creating dirty data table")
        print(e)

# Author: Salvatore
# The function creates all the default tables expected by this system
def create_tables(db_connection):
    cursor=db_connection.cursor()
    try:
        create_user_table(cursor)
        create_prediction_table(cursor)
        create_prediction_model_table(cursor)
        create_package_table(cursor)
        create_training_data_table(cursor)
        create_dirty_data_table(cursor)
    except Error as e:
        print("Error while the tables")
        print(e)

    try:
        db_connection.commit()
    except Error as e:
        print("Error while committing the cursor")
        print(e)

# Author: Salvatore
# Drops all the tables of the db
def drop_tables(db_connection):
    try:
        cursor=db_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")  # necessary to delete on cascade
        dropUserTableStatement="DROP TABLE IF EXISTS user"
        dropPredictionTableStatement="DROP TABLE IF EXISTS prediction"
        dropPredictionModelTableStatement="DROP TABLE IF EXISTS prediction_model"
        dropPackageTableStatement="DROP TABLE IF EXISTS package"
        dropTrainingDataTableStatement="DROP TABLE IF EXISTS training_data"
        dropDirtyDataTableStatement="DROP TABLE IF EXISTS dirty_data"
        dropDF="DROP TABLE IF EXISTS datasource"


        cursor.execute(dropUserTableStatement)
        cursor.execute(dropPredictionModelTableStatement)
        cursor.execute(dropPredictionTableStatement)
        cursor.execute(dropPackageTableStatement)
        cursor.execute(dropTrainingDataTableStatement)
        cursor.execute(dropDirtyDataTableStatement)
        cursor.execute(dropDF)
    except Error as e:
        print("Error while dropping all the tables")

#Author: Salvatore
def insertUser(db_connection, user):
    cursor=db_connection.cursor()
    sql="" \
        "INSERT INTO user(name, email, password, user_type)" \
        "VALUES (?, ?, ?, ?)"
    try:
        cursor.execute(sql, user)
    except Error as e:
        print("Error inserting user")
        print(e)

#Author: Salvatore
def insertPrediction(db_connection, prediction):
    cursor=db_connection.cursor()
    sql="" \
        "INSERT INTO prediction (date, type_predicted, user_id)" \
        "VALUES (?, ?, ?)"
    try:
        cursor.execute(sql, prediction)
    except Error as e:
        print("Error inserting prediction")
        print(e)

#Author: Salvatore
def insertPredictionModel(db_connection, predictionModel):
    cursor=db_connection.cursor()
    sql="" \
        "INSERT INTO prediction_model (accuracy)" \
        "VALUES (?)"
    try:
        cursor.execute(sql, predictionModel)
    except Error as e:
        print("Error inserting prediction model")
        print(e)

#Author: Salvatore
def insertPackage(db_connection, package):
    cursor=db_connection.cursor()
    sql="" \
        "INSERT INTO package (name, version)" \
        "VALUES (?, ?)"
    try:
        cursor.execute(sql, package)
    except Error as e:
        print("Error inserting prediction package")
        print(e)

#Author: Salvatore
def insertTrainingData(db_connection, training_data):
    cursor=db_connection.cursor()
    sql="" \
        "INSERT INTO training_data (training_row, clean_post, type, dirty_row)" \
        "VALUES (?, ?, ?, ?)"
    try:
        cursor.execute(sql, training_data)
    except Error as e:
        print("Error inserting training data")
        print(e)

#Author: Salvatore
def insertDirtyData(db_connection, dirty_data):
    cursor=db_connection.cursor()
    sql="" \
        "INSERT INTO dirty_data (dirty_row, dirty_post, type)" \
        "VALUES (?, ?, ?)"
    try:
        cursor.execute(sql, dirty_data)
    except Error as e:
        print("Error inserting dirty data package")
        print(e)


def initializeTypesTable():
    post_new = Post(
        content=sentence,
        date_posted=datetime.now(),
        user=request.user,
        type_predicted=response,
        predictor=Prediction_model.objects.latest("id")
    )



