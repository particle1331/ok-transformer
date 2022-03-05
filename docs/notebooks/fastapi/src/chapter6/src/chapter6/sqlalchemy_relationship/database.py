from databases import Database
from sqlalchemy import create_engine
import pandas as pd


DATABASE_URL = "sqlite:///chapter6_sqlalchemy.db"
database = Database(DATABASE_URL)
sqlalchemy_engine = create_engine(DATABASE_URL)


def get_database() -> Database:
    return database
