from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

python_code = """
import numpy as np
from typing import List, Optional

def calculate_mean(numbers: List[float]) -> float:
    '''Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        The mean value
    '''
    return sum(numbers) / len(numbers)

def calculate_median(numbers: List[float]) -> float:
    '''Calculate the median of a list of numbers.'''
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    return sorted_nums[mid]

class StatisticalAnalyzer:
    '''A class for performing statistical analysis on datasets.'''
    
    def __init__(self, data: List[float]):
        self.data = data
        self.mean = None
        self.median = None
    
    def analyze(self) -> dict:
        '''Perform complete statistical analysis.'''
        self.mean = calculate_mean(self.data)
        self.median = calculate_median(self.data)
        
        return {
            'mean': self.mean,
            'median': self.median,
            'count': len(self.data)
        }
    
    def get_summary(self) -> str:
        '''Return a formatted summary of the analysis.'''
        if self.mean is None:
            self.analyze()
        
        return f"Mean: {self.mean:.2f}, Median: {self.median:.2f}"

def main():
    '''Main execution function.'''
    data = [1.5, 2.3, 3.7, 4.2, 5.1]
    analyzer = StatisticalAnalyzer(data)
    results = analyzer.analyze()
    print(analyzer.get_summary())

if __name__ == "__main__":
    main()
"""
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=400,
    chunk_overlap=50,
)

code_chunk = code_splitter.split_text(text=python_code)

print(code_chunk)
print(f"no. of chunks: {len(code_chunk)}")

# for chunk size 100
# [
# 'import numpy as np\nfrom typing import List, Optional',
# 'def calculate_mean(numbers: List[float]) -> float:',
# "'''Calculate the arithmetic mean of a list of numbers.\n    \n    Args:",
# 'Args:\n        numbers: List of numerical values\n        \n    Returns:\n        The mean value',
# "'''\n    return sum(numbers) / len(numbers)",
# 'def calculate_median(numbers: List[float]) -> float:',
# "'''Calculate the median of a list of numbers.'''\n    sorted_nums = sorted(numbers)",
# 'n = len(sorted_nums)\n    mid = n // 2\n    \n    if n % 2 == 0:',
# 'return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2\n    return sorted_nums[mid]',
# "class StatisticalAnalyzer:\n    '''A class for performing statistical analysis on datasets.'''",
# 'def __init__(self, data: List[float]):\n        self.data = data\n        self.mean = None',
# 'self.median = None\n    \n    def analyze(self) -> dict:',
# "'''Perform complete statistical analysis.'''\n        self.mean = calculate_mean(self.data)",
# 'self.median = calculate_median(self.data)\n        \n        return {', "'mean': self.mean,\n            'median': self.median,",
# "'count': len(self.data)\n        }\n    \n    def get_summary(self) -> str:",
# "'''Return a formatted summary of the analysis.'''\n        if self.mean is None:",
# 'self.analyze()', 'return f"Mean: {self.mean:.2f},
# Median: {self.median:.2f}"',
# "def main():\n    '''Main execution function.'''\n    data = [1.5, 2.3, 3.7, 4.2, 5.1]",
# 'analyzer = StatisticalAnalyzer(data)\n    results = analyzer.analyze()',
# 'print(analyzer.get_summary())',
# 'if __name__ == "__main__":\n    main()'
# ]
# no. of chunks: 23

# for chunk size 400
# [
# "import numpy as np\nfrom typing import List,
# Optional\n\ndef calculate_mean(numbers: List[float]) -> float:\n    '''Calculate the arithmetic mean of a list of numbers.\n    \n    Args:\n        numbers: List of numerical values\n        \n    Returns:\n        The mean value\n    '''\n    return sum(numbers) / len(numbers)",
# "def calculate_median(numbers: List[float]) -> float:\n    '''Calculate the median of a list of numbers.'''\n    sorted_nums = sorted(numbers)\n    n = len(sorted_nums)\n    mid = n // 2\n    \n    if n % 2 == 0:\n        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2\n    return sorted_nums[mid]",
# "class StatisticalAnalyzer:\n    '''A class for performing statistical analysis on datasets.'''\n    \n    def __init__(self, data: List[float]):\n        self.data = data\n        self.mean = None\n        self.median = None\n    \n    def analyze(self) -> dict:\n        '''Perform complete statistical analysis.'''\n        self.mean = calculate_mean(self.data)",
# "self.mean = calculate_mean(self.data)\n        self.median = calculate_median(self.data)\n        \n        return {\n            'mean': self.mean,\n            'median': self.median,\n            'count': len(self.data)\n        }\n    \n    def get_summary(self) -> str:\n        '''Return a formatted summary of the analysis.'''\n        if self.mean is None:\n            self.analyze()", 'self.analyze()\n        \n        return f"Mean: {self.mean:.2f},
# Median: {self.median:.2f}"', 'def main():\n    \'\'\'Main execution function.\'\'\'\n    data = [1.5, 2.3, 3.7, 4.2, 5.1]\n    analyzer = StatisticalAnalyzer(data)\n    results = analyzer.analyze()\n    print(analyzer.get_summary())\n\nif __name__ == "__main__":\n    main()'
# ]
# no. of chunks: 6

# Separator used
print(code_splitter.get_separators_for_language(language=Language.PYTHON))
# ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# NOTE: when we use from_language() method, we can't use separator arg, coz its already used in func.

# RECURSIVE SPLITTING FOR JSON

JSON_DATA = {
    "company": "AI Research Corp",
    "departments": [
        {
            "name": "Machine Learning",
            "team_size": 25,
            "projects": [
                {
                    "id": "ML001",
                    "title": "Computer Vision System",
                    "description": "Developing advanced image recognition using CNNs",
                    "status": "active",
                    "team_members": ["Alice", "Bob", "Charlie"],
                },
                {
                    "id": "ML002",
                    "title": "NLP Platform",
                    "description": "Building transformer-based language models",
                    "status": "active",
                    "team_members": ["David", "Eve"],
                },
            ],
        },
        {
            "name": "Data Engineering",
            "team_size": 15,
            "projects": [
                {
                    "id": "DE001",
                    "title": "Data Pipeline",
                    "description": "ETL pipeline for real-time data processing",
                    "status": "active",
                }
            ],
        },
    ],
    "technologies": {
        "frameworks": ["TensorFlow", "PyTorch", "scikit-learn"],
        "languages": ["Python", "R", "Julia"],
        "cloud": ["AWS", "Google Cloud", "Azure"],
    },
    "metadata": {"founded": 2020, "headquarters": "San Francisco", "employees": 150},
}

json_splitter = RecursiveJsonSplitter(max_chunk_size=400)

json_chunk = json_splitter.split_json(json_data=JSON_DATA)

print(json_chunk)
print(f"no. of chunks: {len(json_chunk)}")

# [
# {'company': 'AI Research Corp', 'departments': [{'name': 'Machine Learning', 'team_size': 25, 'projects': [{'id': 'ML001', 'title': 'Computer Vision System', 'description': 'Developing advanced image recognition using CNNs', 'status': 'active', 'team_members': ['Alice', 'Bob', 'Charlie']}, {'id': 'ML002', 'title': 'NLP Platform', 'description': 'Building transformer-based language models', 'status': 'active', 'team_members': ['David', 'Eve']}]}, {'name': 'Data Engineering', 'team_size': 15, 'projects': [{'id': 'DE001', 'title': 'Data Pipeline', 'description': 'ETL pipeline for real-time data processing', 'status': 'active'}]}]},
# {'technologies': {'frameworks': ['TensorFlow', 'PyTorch', 'scikit-learn'], 'languages': ['Python', 'R', 'Julia'], 'cloud': ['AWS', 'Google Cloud', 'Azure']}, 'metadata': {'founded': 2020, 'headquarters': 'San Francisco', 'employees': 150}}
# ]
# no. of chunks: 2

json_text = json_splitter.split_text(json_data=JSON_DATA)

print(json_text)
print(f"no. of chunks: {len(json_text)}")

# [
# '{"company": "AI Research Corp", "departments": [{"name": "Machine Learning", "team_size": 25, "projects": [{"id": "ML001", "title": "Computer Vision System", "description": "Developing advanced image recognition using CNNs", "status": "active", "team_members": ["Alice", "Bob", "Charlie"]}, {"id": "ML002", "title": "NLP Platform", "description": "Building transformer-based language models", "status": "active", "team_members": ["David", "Eve"]}]}, {"name": "Data Engineering", "team_size": 15, "projects": [{"id": "DE001", "title": "Data Pipeline", "description": "ETL pipeline for real-time data processing", "status": "active"}]}]}',
# '{"technologies": {"frameworks": ["TensorFlow", "PyTorch", "scikit-learn"], "languages": ["Python", "R", "Julia"], "cloud": ["AWS", "Google Cloud", "Azure"]}, "metadata": {"founded": 2020, "headquarters": "San Francisco", "employees": 150}}'
# ]
# no. of chunks: 2

# MARKDOWN SPLITTING

MARKDOWN_TEXT = """# Artificial Intelligence Overview

Artificial intelligence is transforming technology and shaping the future of computing.

## Machine Learning

Machine learning is a subset of AI that focuses on pattern recognition.

### Supervised Learning

Supervised learning algorithms learn from labeled training data.
They make predictions based on input-output pairs.

Common algorithms include:
- Linear regression
- Decision trees
- Support vector machines

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.
It's useful for clustering and dimensionality reduction.

Common techniques:
- K-means clustering
- Principal component analysis
- Hierarchical clustering

## Deep Learning

Deep learning uses neural networks with multiple layers.

### Neural Networks

Neural networks are inspired by biological neurons.
They consist of interconnected nodes organized in layers.

### Convolutional Neural Networks

CNNs excel at image recognition tasks.
They use convolutional layers to detect features hierarchically.

## Applications

AI has applications across multiple domains:

### Healthcare

- Disease diagnosis
- Drug discovery
- Medical imaging analysis

### Finance

- Fraud detection
- Algorithmic trading
- Risk assessment
"""

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "header_1"), ("##", "header_2")], strip_headers=False
)

md_chunk = md_splitter.split_text(text=MARKDOWN_TEXT)

print(md_chunk)
print(f"no. of chunks: {len(md_chunk)}")

# [
# Document(metadata={'header_1': 'Artificial Intelligence Overview'}, page_content='# Artificial Intelligence Overview  \nArtificial intelligence is transforming technology and shaping the future of computing.'),
# Document(metadata={'header_1': 'Artificial Intelligence Overview', 'header_2': 'Machine Learning'}, page_content="## Machine Learning  \nMachine learning is a subset of AI that focuses on pattern recognition.  \n### Supervised Learning  \nSupervised learning algorithms learn from labeled training data.\nThey make predictions based on input-output pairs.  \nCommon algorithms include:\n- Linear regression\n- Decision trees\n- Support vector machines  \n### Unsupervised Learning  \nUnsupervised learning finds patterns in unlabeled data.\nIt's useful for clustering and dimensionality reduction.  \nCommon techniques:\n- K-means clustering\n- Principal component analysis\n- Hierarchical clustering"),
# Document(metadata={'header_1': 'Artificial Intelligence Overview', 'header_2': 'Deep Learning'}, page_content='## Deep Learning  \nDeep learning uses neural networks with multiple layers.  \n### Neural Networks  \nNeural networks are inspired by biological neurons.\nThey consist of interconnected nodes organized in layers.  \n### Convolutional Neural Networks  \nCNNs excel at image recognition tasks.\nThey use convolutional layers to detect features hierarchically.'),
# Document(metadata={'header_1': 'Artificial Intelligence Overview', 'header_2': 'Applications'}, page_content='## Applications  \nAI has applications across multiple domains:  \n### Healthcare  \n- Disease diagnosis\n- Drug discovery\n- Medical imaging analysis  \n### Finance  \n- Fraud detection\n- Algorithmic trading\n- Risk assessment')]
# no. of chunks: 4
