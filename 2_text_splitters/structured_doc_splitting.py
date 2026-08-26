from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

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
