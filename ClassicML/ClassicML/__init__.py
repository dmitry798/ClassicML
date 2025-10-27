"""ClassicML - Machine Learning from scratch in C++"""

import numpy as np

__version__ = "1.0.0"

try:
    # Импортируем C++ модуль
    from ._classicml import (
        Matrix as _Matrix,
        Dataset as _Dataset,
        StandardScaler,
        Errors,
        LinearRegression,
        LogisticRegression,
        Knn,
        KnnRegression,
        sigmoid,
        softmax,
        one_hot_encoder,
        decoder_ohe,
        mean,
        stddev,
        mode,
        sort,
    )
    
    # ========== Wrapper для Matrix с автоматической конвертацией ==========
    class Matrix(_Matrix):
        """Matrix class with automatic NumPy conversion"""
        
        def __init__(self, *args, **kwargs):
            """
            Create Matrix from various inputs:
            - Matrix(rows, cols, name="")
            - Matrix(numpy_array, name="")
            - Matrix()  # empty
            """
            if len(args) == 0:
                # Default constructor
                super().__init__()
                
            elif len(args) == 1:
                arg = args[0]
                
                # Проверка на _Matrix ПЕРВАЯ!
                if isinstance(arg, _Matrix):
                    # Copy constructor
                    super().__init__(arg)
                    
                elif isinstance(arg, np.ndarray):
                    # NumPy array constructor
                    arr = arg
                    name = kwargs.get('name', '')
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    elif arr.ndim != 2:
                        raise ValueError("NumPy array must be 1D or 2D")
                    
                    rows, cols = arr.shape
                    super().__init__(rows, cols, name)
                    
                    # Fill with data
                    arr = np.ascontiguousarray(arr, dtype=np.float64)
                    for i in range(rows):
                        for j in range(cols):
                            self[i * cols + j] = float(arr[i, j])
                else:
                    raise ValueError(f"Cannot create Matrix from type {type(arg)}")
                    
            elif len(args) == 2:
                # Может быть: (rows, cols) или (numpy_array, name)
                if isinstance(args[0], int) and isinstance(args[1], int):
                    # (rows, cols)
                    super().__init__(args[0], args[1], kwargs.get('name', ''))
                elif isinstance(args[0], np.ndarray) and isinstance(args[1], str):
                    # (numpy_array, name)
                    arr = args[0]
                    name = args[1]
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    elif arr.ndim != 2:
                        raise ValueError("NumPy array must be 1D or 2D")
                    
                    rows, cols = arr.shape
                    super().__init__(rows, cols, name)
                    
                    # Fill with data
                    arr = np.ascontiguousarray(arr, dtype=np.float64)
                    for i in range(rows):
                        for j in range(cols):
                            self[i * cols + j] = float(arr[i, j])
                else:
                    raise ValueError(f"Invalid arguments: {type(args[0])}, {type(args[1])}")
                    
            elif len(args) == 3:
                # (rows, cols, name)
                if isinstance(args[0], int) and isinstance(args[1], int) and isinstance(args[2], str):
                    super().__init__(args[0], args[1], args[2])
                else:
                    raise ValueError(f"Expected (int, int, str), got ({type(args[0])}, {type(args[1])}, {type(args[2])})")
            else:
                raise ValueError(f"Invalid number of arguments: {len(args)}")
        
        def to_numpy(self):
            """Convert Matrix to NumPy array"""
            rows, cols = self.get_rows(), self.get_cols()
            arr = np.zeros((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    arr[i, j] = self(i, j)
            return arr
    
    # ========== Wrapper для Dataset с автоматической конвертацией ==========
    class Dataset(_Dataset):
        """Dataset class with automatic NumPy conversion"""
        
        def __init__(self, X, Y):
            """
            Create Dataset from X and Y
            
            Parameters:
                X: Matrix or NumPy array (features)
                Y: Matrix or NumPy array (targets)
            """
            # Конвертируем NumPy в Matrix если нужно
            if isinstance(X, np.ndarray):
                X = Matrix(X, "X")
            if isinstance(Y, np.ndarray):
                Y = Matrix(Y, "Y")
            
            # Вызываем оригинальный конструктор
            super().__init__(X, Y)
    
    # Aliases для удобства
    KNNClassifier = Knn
    KNNRegressor = KnnRegression
    
    # Экспортируем всё
    __all__ = [
        '__version__',
        'Matrix', 'Dataset', 'StandardScaler', 'Errors',
        'LinearRegression', 'LogisticRegression', 
        'Knn', 'KnnRegression',
        'KNNClassifier', 'KNNRegressor',
        'sigmoid', 'softmax', 'one_hot_encoder', 'decoder_ohe',
        'mean', 'stddev', 'mode', 'sort',
    ]

except ImportError as e:
    print(f"ERROR: Could not import C++ module: {e}")
    print("The library is not built. Run: pip install -e .")
    raise
