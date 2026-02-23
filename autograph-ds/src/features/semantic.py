"""Semantic fingerprinting features for authorship attribution.

This module extracts high-level semantic patterns that capture:
- Import/library usage signatures
- API call patterns and function/method preferences  
- Code domain/topic fingerprints
"""

import re
from typing import Dict, Any, List, Set, Tuple
from collections import Counter
from tree_sitter import Node


# Top 50 most common Python libraries for fingerprinting
TOP_LIBRARIES = [
    # Standard library (most common)
    'os', 'sys', 'json', 'collections', 'itertools', 'functools', 'datetime', 
    're', 'math', 'random', 'typing', 'pathlib', 'inspect', 'hashlib',
    'logging', 'unittest', 'threading', 'multiprocessing', 'subprocess',
    'io', 'csv', 'pickle', 'copy', 'enum', 'contextlib', 'dataclasses',
    'abc', 'numbers', 'string', 'textwrap', 'warnings', 'types', 'decimal',
    'statistics', 'zoneinfo', 'zoneinfo', 'tomllib', 'asyncio',
    # Third-party (web/HTTP)
    'requests', 'httpx', 'urllib', 'flask', 'django', 'fastapi', 'tornado',
    # Data/ML
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow',
    'torch', 'keras', 'xgboost', 'lightgbm', 'scipy', 'plotly', 'altair',
    # Testing
    'pytest', 'mock', 'hypothesis', 'factory_boy', 'faker',
    # Utilities
    'click', 'typer', 'argparse', 'rich', 'pydantic', 'attrs', 'structlog',
    # Database
    'sqlalchemy', 'sqlite3', 'psycopg2', 'pymongo', 'redis',
    # Async
    'aiohttp', 'aiobotocore', 'trio', 'anyio',
    # Other common
    'jinja2', 'yaml', 'toml', 'pillow', 'boto3', 'botocore', 'requests',
    'beautifulsoup4', 'lxml', 'selenium', 'playwright', 'twisted', 'zope',
]

# Domain categories for semantic clustering
LIBRARY_DOMAINS = {
    'web': ['requests', 'httpx', 'urllib', 'flask', 'django', 'fastapi', 'tornado', 
            'aiohttp', 'jinja2', 'beautifulsoup4', 'lxml', 'selenium', 'playwright'],
    'data_science': ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'scipy',
                     'plotly', 'altair', 'xgboost', 'lightgbm', 'statsmodels'],
    'ml_ai': ['tensorflow', 'torch', 'keras', 'transformers', 'sklearn', 'jax',
              'mlflow', 'wandb', 'tensorboard'],
    'testing': ['pytest', 'unittest', 'mock', 'hypothesis', 'factory_boy', 'faker'],
    'cli_tools': ['click', 'typer', 'argparse', 'rich', 'colorama', 'progress', 
                  'tqdm', 'prompt_toolkit'],
    'database': ['sqlalchemy', 'sqlite3', 'psycopg2', 'pymongo', 'redis', 
                 'peewee', 'django', 'alembic'],
    'async': ['asyncio', 'aiohttp', 'aiobotocore', 'trio', 'anyio', 'curio'],
    'systems': ['os', 'sys', 'subprocess', 'pathlib', 'shutil', 'signal', 'mmap'],
    'dev_tools': ['pydantic', 'attrs', 'dataclasses', 'typing', 'mypy', 'black',
                  'flake8', 'pylint', 'bandit'],
    'cloud': ['boto3', 'botocore', 'google', 'azure', 'kubernetes', 'docker'],
    'serialization': ['json', 'pickle', 'yaml', 'toml', 'msgpack', 'protobuf', 
                      'avro', 'xml'],
    'crypto_security': ['hashlib', 'secrets', 'cryptography', 'bcrypt', 'jwt',
                        'openssl', 'pynacl'],
    'text_processing': ['re', 'string', 'textwrap', 'difflib', 'fuzzywuzzy',
                        'rapidfuzz', 'regex', ' Levenshtein'],
    'concurrency': ['threading', 'multiprocessing', 'concurrent', 'queue', 'asyncio'],
    'networking': ['socket', 'ssl', 'http', 'ftplib', 'smtplib', 'imaplib', 'urllib'],
}


class SemanticFingerprintExtractor:
    """Extracts semantic fingerprints from Python code."""
    
    def __init__(self):
        self.top_libraries = TOP_LIBRARIES[:50]  # Top 50 for fingerprinting
        self.library_domains = LIBRARY_DOMAINS
    
    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        """Extract all semantic fingerprint features."""
        features = {}
        
        # Import fingerprinting
        import_features = self._extract_import_fingerprint(code, root_node)
        features.update(import_features)
        
        # API call analysis
        api_features = self._extract_api_calls(code, root_node)
        features.update(api_features)
        
        return features
    
    def _extract_import_fingerprint(self, code: str, root_node: Node) -> Dict[str, Any]:
        """
        Extract import fingerprint features (50+ features).
        
        Creates a signature based on which libraries are imported and how.
        """
        features = {}
        
        # Parse imports from AST
        imported_libs = set()
        import_froms = set()
        import_styles = Counter()  # 'import x', 'from x import y', 'import x as y'
        
        def traverse_imports(node: Node):
            if node.type == 'import_statement':
                # Direct import: import foo, import foo.bar
                names = self._get_import_names(node)
                for name in names:
                    base_lib = name.split('.')[0]
                    imported_libs.add(base_lib)
                    import_styles['direct'] += 1
                    
            elif node.type == 'import_from_statement':
                # From import: from foo import bar
                module = self._get_from_module(node)
                if module:
                    base_lib = module.split('.')[0]
                    imported_libs.add(base_lib)
                    import_froms.add(base_lib)
                    import_styles['from'] += 1
                    
                    # Check for 'import *' pattern
                    if self._has_star_import(node):
                        import_styles['star'] += 1
            
            for child in node.children:
                traverse_imports(child)
        
        traverse_imports(root_node)
        
        # Create binary feature vector for top 50 libraries (1 = imported, 0 = not)
        for lib in self.top_libraries:
            features[f'uses_{lib}'] = 1 if lib in imported_libs else 0
        
        # Domain usage ratios
        total_libs = len(imported_libs) if imported_libs else 1
        for domain, libs in self.library_domains.items():
            domain_count = sum(1 for lib in libs if lib in imported_libs)
            features[f'{domain}_domain_ratio'] = domain_count / total_libs
        
        # Import style preferences
        total_imports = sum(import_styles.values()) if import_styles else 1
        features['direct_import_ratio'] = import_styles.get('direct', 0) / total_imports
        features['from_import_ratio'] = import_styles.get('from', 0) / total_imports
        features['star_import_ratio'] = import_styles.get('star', 0) / total_imports
        
        # Standard library vs third-party ratio
        stdlib_libs = {'os', 'sys', 'json', 'collections', 'itertools', 'functools',
                      'datetime', 're', 'math', 'random', 'typing', 'pathlib', 
                      'inspect', 'hashlib', 'logging', 'unittest', 'threading',
                      'multiprocessing', 'subprocess', 'io', 'csv', 'pickle',
                      'copy', 'enum', 'contextlib', 'dataclasses', 'abc',
                      'numbers', 'string', 'textwrap', 'warnings', 'types',
                      'decimal', 'statistics', 'zoneinfo', 'tomllib', 'asyncio',
                      'sqlite3', 'socket', 'ssl', 'http', 'ftplib', 'smtplib',
                      'imaplib', 'urllib', 'queue', 'concurrent', 'mmap', 'signal'}
        
        stdlib_count = sum(1 for lib in imported_libs if lib in stdlib_libs)
        third_party_count = len(imported_libs) - stdlib_count
        total = len(imported_libs) if imported_libs else 1
        
        features['stdlib_ratio'] = stdlib_count / total
        features['third_party_ratio'] = third_party_count / total
        features['unique_library_count'] = len(imported_libs)
        
        return features
    
    def _extract_api_calls(self, code: str, root_node: Node) -> Dict[str, Any]:
        """
        Extract API call pattern features (30 features).
        
        Analyzes which functions/methods are called and how.
        """
        features = {}
        
        # Collect all call expressions
        function_calls = []
        method_calls = []
        attribute_accesses = []
        
        def traverse_calls(node: Node, in_call: bool = False):
            if node.type == 'call':
                # Get the function being called
                func_node = node.child_by_field_name('function')
                if func_node:
                    func_name = self._get_node_text(func_node, code)
                    if func_name:
                        function_calls.append(func_name)
                        
                        # Check if it's a method call (has dot)
                        if '.' in func_name:
                            method_calls.append(func_name)
            
            elif node.type == 'attribute':
                # Attribute access (potential method call target)
                attr_name = self._get_node_text(node, code)
                if attr_name:
                    attribute_accesses.append(attr_name)
            
            for child in node.children:
                traverse_calls(child, in_call=(in_call or node.type == 'call'))
        
        traverse_calls(root_node)
        
        # Top 20 most common function/method calls (normalized frequencies)
        call_counter = Counter(function_calls)
        top_calls = call_counter.most_common(20)
        
        total_calls = len(function_calls) if function_calls else 1
        for func_name, count in top_calls:
            # Sanitize function name for feature key
            safe_name = re.sub(r'[^\w]', '_', func_name)[:40]
            features[f'call_freq_{safe_name}'] = count / total_calls
        
        # Fill remaining with 0 if less than 20 unique calls
        for i in range(len(top_calls), 20):
            features[f'call_freq_placeholder_{i}'] = 0.0
        
        # Call style metrics
        total_method_calls = len(method_calls)
        features['method_call_ratio'] = total_method_calls / total_calls if total_calls > 0 else 0
        
        # Chained method calls (e.g., df.groupby().agg().reset_index())
        chained_calls = len([c for c in function_calls if c.count('.') > 1])
        features['chained_call_ratio'] = chained_calls / total_calls if total_calls > 0 else 0
        
        # Common patterns
        common_patterns = {
            'print_calls': len([c for c in function_calls if c == 'print']),
            'len_calls': len([c for c in function_calls if c == 'len']),
            'range_calls': len([c for c in function_calls if c == 'range']),
            'enumerate_calls': len([c for c in function_calls if c == 'enumerate']),
            'zip_calls': len([c for c in function_calls if c == 'zip']),
            'isinstance_calls': len([c for c in function_calls if c == 'isinstance']),
            'hasattr_calls': len([c for c in function_calls if c == 'hasattr']),
            'getattr_calls': len([c for c in function_calls if c == 'getattr']),
            'open_calls': len([c for c in function_calls if c == 'open']),
            'join_calls': len([c for c in function_calls if c.endswith('.join')]),
        }
        
        for pattern, count in common_patterns.items():
            features[f'{pattern}_ratio'] = count / total_calls if total_calls > 0 else 0
        
        # Argument passing style (analyze call argument_list nodes)
        keyword_args = len(re.findall(r'\b\w+\s*=', code))
        positional_args = len(function_calls)  # Simplified estimate
        total_args = keyword_args + positional_args
        
        features['keyword_arg_ratio'] = keyword_args / total_args if total_args > 0 else 0.5
        
        # Unique functions called (vocabulary richness)
        features['call_vocabulary_size'] = len(set(function_calls))
        features['call_diversity'] = len(set(function_calls)) / total_calls if total_calls > 0 else 0
        
        return features
    
    def _get_import_names(self, node: Node) -> List[str]:
        """Extract imported names from import_statement node."""
        names = []
        for child in node.children:
            if child.type == 'dotted_name':
                name = self._get_dotted_name_text(child)
                if name:
                    names.append(name)
            elif child.type == 'aliased_import':
                # import x as y - get the original name
                original = child.child_by_field_name('name')
                if original and original.type == 'dotted_name':
                    name = self._get_dotted_name_text(original)
                    if name:
                        names.append(name)
        return names
    
    def _get_from_module(self, node: Node) -> str:
        """Extract module name from import_from_statement."""
        for child in node.children:
            if child.type == 'dotted_name':
                return self._get_dotted_name_text(child)
        return None
    
    def _has_star_import(self, node: Node) -> bool:
        """Check if import_from_statement uses wildcard import."""
        for child in node.children:
            if child.type == 'wildcard_import':
                return True
        return False
    
    def _get_dotted_name_text(self, node: Node) -> str:
        """Extract text from dotted_name node."""
        parts = []
        for child in node.children:
            if child.type == 'identifier':
                parts.append(child.text.decode('utf-8') if isinstance(child.text, bytes) else child.text)
            elif child.type == '.':
                parts.append('.')
        return ''.join(parts)
    
    def _get_node_text(self, node: Node, code: str) -> str:
        """Extract text from a node."""
        try:
            start = node.start_byte
            end = node.end_byte
            return code[start:end]
        except:
            return ""
