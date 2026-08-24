from pathlib import Path
from pprint import pp

from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.pdf import PDFMinerLoader, PyPDFLoader

# define path
path = Path("./knowledge_source/attention_is_all_you_need.pdf")

# PyPDF LOADER
pypdf_loader = PyPDFLoader(
    file_path=path.as_posix(), mode="page"
)  # as posix return str representation of path with / slashes

# load docs
pypdf_docs = pypdf_loader.load()

# pp(pypdf_docs)

# [Document(metadata={'producer': 'pdfTeX-1.40.25', 'creator': 'LaTeX with hyperref', 'creationdate': '2024-04-10T21:11:43+00:00', 'author': '', 'keywords': '', 'moddate': '2024-04-10T21:11:43+00:00', 'ptex.fullbanner': 'This is pdfTeX, Version 3.141592653-2.6-1.40.25 (TeX Live 2023) kpathsea version 6.3.5', 'subject': '', 'title': '', 'trapped': '/False', 'source': 'knowledge_source/attention_is_all_you_need.pdf', 'total_pages': 15, 'page': 0, 'page_label': '1'}, page_content='Provided proper attribution is provided, Google hereby grants permission to\nreproduce the tables and figures in this paper solely for use in journalistic or\nscholarly works.\nAttention Is All You Need\nAshish Vaswani∗\nGoogle Brain\navaswani@google.com\nNoam Shazeer∗\nGoogle Brain\nnoam@google.com\nNiki Parmar∗\nGoogle Research\nnikip@google.com\nJakob Uszkoreit∗\nGoogle Research\nusz@google.com\nLlion Jones∗\nGoogle Research\nllion@google.com\nAidan N. Gomez∗†\nUniversity of Toronto\naidan@cs.toronto.edu\nŁukasz Kaiser∗\nGoogle Brain\nlukaszkaiser@google.com\nIllia Polosukhin∗‡\nillia.polosukhin@gmail.com\nAbstract\nThe dominant sequence transduction models are based on complex recurrent or\nconvolutional neural networks that include an encoder and a decoder. The best\nperforming models also connect the encoder and decoder through an attention\nmechanism. We propose a new simple network architecture, the Transformer,\nbased solely on attention mechanisms, dispensing with recurrence and convolutions\nentirely. Experiments on two machine translation tasks show these models to\nbe superior in quality while being more parallelizable and requiring significantly\nless time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-\nto-German translation task, improving over the existing best results, including\nensembles, by over 2 BLEU. On...

# print no. of docs
# pp(len(pypdf_docs))  # 15 docs for 15 page pdf

# print first page content
# pp(pypdf_docs[2].page_content)

# ('Provided proper attribution is provided, Google hereby grants permission to\n'
#  'reproduce the tables and figures in this paper solely for use in '
#  'journalistic or\n'
#  'scholarly works.\n'
#  'Attention Is All You Need\n'
#  'Ashish Vaswani∗\n'
#  'Google Brain\n'
#  'avaswani@google.com\n'
#  'Noam Shazeer∗\n'
#  'Google Brain\n'
#  'noam@google.com\n'

# PyPDF WITH IMAGE LOADERS (NOTE: it doesn't work well with images use pdfminer instead, this is for demo purpose only)
# pypdf_image_loader = PyPDFLoader(              <---- this doesn't worked
#     file_path=path.as_posix(),
#     mode="page",
#     extract_images=True,
#     images_parser=TesseractBlobParser(),
#     images_inner_format="html-img",
# )

# doc_with_images = pypdf_image_loader.load()

# pp(doc_with_images[2].page_content)

# PDFMINER
pdfminer = PDFMinerLoader(
    file_path=path.as_posix(),
    mode="page",
    extract_images=True,
    images_parser=RapidOCRBlobParser(),
    images_inner_format="html-img",
)

doc_with_images = pdfminer.load()

pp(doc_with_images[2].page_content)

# ...
# 'computed as a weighted sum\n'
#  '\n'
#  '3\n'
#  '\n'
#  '<img alt="Output\n'
#  'Probabilities\n'
#  'Softmax\n'
#  'Linear\n'
#  'Add &amp; Norm\n'
# ...
