from pathlib import Path
from pprint import pp

from langchain_community.document_loaders.csv_loader import CSVLoader

"""
Notes:
    in CSV document, the no. of docs = no. of rows in csv
    you can select which columns should go in page content and in metadata by setting CSVLoader() params
"""

csv_path = Path("./knowledge_source/organizations.csv")

loader = CSVLoader(file_path=csv_path)

documents = loader.load()

pp(documents)
