from pathlib import Path
from pprint import pp

from langchain_community.document_loaders.json_loader import JSONLoader

"""
 "products": [
    {
      "productID": "0000001",
      "manufacturer": "Zara",
      "img": "https://static.zara.net/photos///2023/I/0/2/p/5320/355/800/2/w/563/5320355800_1_1_1.jpg?ts=1697787915583",
      "Url": "https://www.zara.com/in/en/man-outerwear-l715.html",
      "productName": "PINSTRIPE COAT",
      "Description": "Oversize-fit coat made of a viscose blend fabric. Notch lapel collar and long sleeves with buttoned cuffs.",
      "price": 4900,
      "category": "Men Cloths"
    },
    {
      "productID": "0000002",
      "manufacturer": "Zara",
"""

json_path = Path("./knowledge_source/apparels.json")


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["product_name"] = record["productName"]
    metadata["category"] = record["category"]
    metadata["price"] = record["price"]
    # del metadata["seq_num"]                   <----- it will remove seq_num from metadata
    return metadata


json_loader = JSONLoader(
    file_path=json_path.as_posix(),
    jq_schema=".products[]",
    content_key="Description",  # content needed in doc
    metadata_func=metadata_func,  # metadata needed in doc
)

documents = json_loader.load()

pp(documents)

# [Document(metadata={'source': 'D:\\AI\\RAG\\document_loaders\\knowledge_source\\apparels.json', 'seq_num': 1, 'product_name': 'PINSTRIPE COAT', 'category': 'Men Cloths', 'price': 4900}, page_content='Oversize-fit coat made of a viscose blend fabric. Notch lapel collar and long sleeves with buttoned cuffs.'),
#  Document(metadata={'source': 'D:\\AI\\RAG\\document_loaders\\knowledge_source\\apparels.json', 'seq_num': 2, 'product_name': 'OVERSIZED TECHNICAL TRENCH COAT', 'category': 'Men Cloths', 'price': 4900}, page_content='Trench coat made of technical fabric with a velvety finish. Notch lapel collar and long sleeves.'),
#  Document(metadata={'source': 'D:\\AI\\RAG\\document_loaders\\knowledge_source\\apparels.json'...
