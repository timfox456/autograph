# Logical DNA Specification

Logical DNA (L-DNA) is a feature-based representation of code style and structure. It is designed to be extractable via static analysis and useful for probabilistic attribution.

## Feature Buckets

### 1. Structural Topology
Focuses on the "shape" of the code's logic.
- **Node Type Distribution**: Frequency of different AST node types (e.g., `if_statement`, `for_statement`, `call`).
- **Nesting Depth**: Maximum and average depth of nested blocks.
- **Branching Factor**: Average number of children per non-leaf AST node.
- **Complexity Density**: Total nodes divided by lines of code.

### 2. Micro-Stylistics
Focuses on the "how" of the code's formatting.
- **Indentation Style**: Preference for spaces vs. tabs and the standard indentation width.
- **Quote Preference**: Usage of single (`'`) vs. double (`"`) quotes, including triple-quote preferences for docstrings.
- **Trailing Commas**: Frequency of trailing commas in multi-line lists, dictionaries, and function calls.
- **Whitespace Patterns**: Spacing around operators and after commas (planned).

### 3. Logical Idioms
Focuses on the "choice" of language constructs.
- **Naming Conventions**: Ratio of `snake_case` vs. `camelCase` vs. `PascalCase`.
- **String Formatting**: Preference for `f-strings`, `.format()`, or `%` interpolation.
- **Comprehensions**: Frequency and complexity of list, set, and dictionary comprehensions.
- **Async Usage**: Patterns of asynchronous code blocks (planned).

## Implementation

The current implementation uses `tree-sitter` for robust AST parsing of Python code. Features are normalized and flattened into a vector for consumption by machine learning models.

### Privacy Preservation
By selecting only specific buckets, developers can "opt-in" to higher levels of attribution. 
- **High Privacy**: Only share Structural Topology.
- **Full Attribution**: Share all three buckets.
