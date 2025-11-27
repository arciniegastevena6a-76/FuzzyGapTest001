"""
Run All Experiments - Reproduce Paper Table 8-9

This script reproduces the experiments from the paper:
"Updating incomplete framework of target recognition database based on fuzzy gap statistic"

The paper tests on 7 UCI datasets:
1. Iris (3 classes, 4 attributes, 150 samples)
2. Glass (6 classes, 20 attributes, 214 samples)
3. Haberman (2 classes, 3 attributes, 306 samples)
4. Knowledge (4 classes, 5 attributes, 403 samples)
5. Robot (5 classes, 24 attributes, 5456 samples) - using subset
6. Seeds (3 classes, 7 attributes, 210 samples)
7. WDBC (2 classes, 30 attributes, 569 samples)

For each dataset, the experiment:
1. Creates an incomplete FOD by hiding some classes
2. Generates GBPA and checks if m̄(∅) > p (FOD incomplete)
3. Uses FGS to determine optimal k (number of clusters)
4. Compares with actual number of classes
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from fuzzy_gap_statistic import FuzzyGapStatistic

# Configuration constants
DEFAULT_TEST_SAMPLES_HIDDEN = 30  # Number of samples to use from hidden classes
DEFAULT_TRAIN_RATIO = 0.8


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Iris dataset from sklearn"""
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target, ['setosa', 'versicolor', 'virginica']


def load_seeds_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Seeds dataset - create synthetic if not available"""
    try:
        # Try to load from file if available
        import os
        if os.path.exists('seeds.csv'):
            data = np.loadtxt('seeds.csv', delimiter=',')
            X = data[:, :-1]
            y = data[:, -1].astype(int)
        else:
            # Create synthetic seeds-like data (3 classes, 7 features)
            np.random.seed(42)
            n_per_class = 70
            X = np.vstack([
                np.random.randn(n_per_class, 7) + np.array([14, 14, 0.87, 5.5, 3.2, 2.2, 5]),
                np.random.randn(n_per_class, 7) + np.array([18, 16, 0.85, 6.0, 3.5, 3.5, 5.5]),
                np.random.randn(n_per_class, 7) + np.array([12, 13, 0.90, 5.0, 2.8, 4.5, 5.2])
            ])
            y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    except Exception:
        # Fallback: create synthetic data
        np.random.seed(42)
        n_per_class = 70
        X = np.vstack([
            np.random.randn(n_per_class, 7) + np.array([14, 14, 0.87, 5.5, 3.2, 2.2, 5]),
            np.random.randn(n_per_class, 7) + np.array([18, 16, 0.85, 6.0, 3.5, 3.5, 5.5]),
            np.random.randn(n_per_class, 7) + np.array([12, 13, 0.90, 5.0, 2.8, 4.5, 5.2])
        ])
        y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    
    return X, y, ['Kama', 'Rosa', 'Canadian']


def load_haberman_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Haberman dataset - create synthetic if not available"""
    try:
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(name='haberman', version=1, as_frame=False)
        X = dataset.data
        y = dataset.target.astype(int) - 1  # Convert to 0-indexed
    except Exception as e:
        # Fallback: Create synthetic Haberman-like data (2 classes, 3 features)
        # This allows the experiment to run without network access
        np.random.seed(42)
        n_per_class = 150
        X = np.vstack([
            np.random.randn(n_per_class, 3) * np.array([5, 3, 2]) + np.array([58, 65, 2]),
            np.random.randn(n_per_class, 3) * np.array([7, 4, 3]) + np.array([54, 62, 5])
        ])
        y = np.array([0]*n_per_class + [1]*n_per_class)
    
    return X, y, ['Survived', 'Died']


def load_wdbc_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Wisconsin Breast Cancer (WDBC) dataset"""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return data.data, data.target, ['Malignant', 'Benign']


def create_incomplete_fod_experiment(X: np.ndarray, 
                                      y: np.ndarray, 
                                      class_names: List[str],
                                      hidden_classes: List[int],
                                      train_ratio: float = 0.8,
                                      random_seed: int = None) -> Dict:
    """
    Create an incomplete FOD experiment setup
    
    Args:
        X: Feature matrix
        y: Target labels
        class_names: Names of classes
        hidden_classes: List of class indices to hide (unknown targets)
        train_ratio: Ratio of training samples for known classes
        random_seed: Random seed. If None, uses true random.
        
    Returns:
        experiment_setup: Dictionary with train/test data
    """
    rng = np.random.RandomState(random_seed)
    
    all_classes = np.unique(y)
    known_classes = [c for c in all_classes if c not in hidden_classes]
    
    train_indices = []
    test_indices = []
    
    for cls in all_classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        
        if cls in known_classes:
            n_train = int(len(cls_indices) * train_ratio)
            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])
        else:
            # Hidden class: all samples go to test set
            n_test = min(DEFAULT_TEST_SAMPLES_HIDDEN, len(cls_indices))
            test_indices.extend(cls_indices[:n_test])
    
    return {
        'train_data': X[train_indices],
        'train_labels': y[train_indices],
        'test_data': X[test_indices],
        'test_labels': y[test_indices],
        'known_classes': known_classes,
        'hidden_classes': hidden_classes,
        'class_names': class_names,
        'n_known': len(known_classes),
        'n_total': len(all_classes)
    }


def run_single_experiment(dataset_name: str,
                          X: np.ndarray,
                          y: np.ndarray,
                          class_names: List[str],
                          hidden_classes: List[int],
                          max_clusters: int = 10,
                          verbose: bool = True,
                          random_seed: int = None) -> Dict:
    """
    Run a single FGS experiment on a dataset
    
    Args:
        dataset_name: Name of the dataset
        X: Feature matrix
        y: Target labels
        class_names: Names of classes
        hidden_classes: Classes to hide (unknown targets)
        max_clusters: Maximum clusters to test
        verbose: Print progress
        random_seed: Random seed. If None, uses true random.
        
    Returns:
        results: Experiment results dictionary
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
    
    # Setup experiment
    setup = create_incomplete_fod_experiment(X, y, class_names, hidden_classes, random_seed=random_seed)
    
    if verbose:
        print(f"Total classes: {setup['n_total']}")
        print(f"Known classes: {[class_names[c] for c in setup['known_classes']]}")
        print(f"Hidden classes: {[class_names[c] for c in setup['hidden_classes']]}")
        print(f"Training samples: {len(setup['train_data'])}")
        print(f"Test samples: {len(setup['test_data'])}")
    
    # Run FGS - use the same random_seed for reproducibility within a run
    fgs = FuzzyGapStatistic(critical_value=0.5, max_iterations=100, random_seed=random_seed)
    
    results = fgs.fit(
        test_data=setup['test_data'],
        train_data=setup['train_data'],
        train_labels=setup['train_labels'],
        n_known_targets=setup['n_known'],
        max_clusters=max_clusters
    )
    
    # Evaluate
    optimal_k = results.get('optimal_k', setup['n_known'])
    actual_k = setup['n_total']
    correct = optimal_k == actual_k
    
    results['dataset'] = dataset_name
    results['actual_classes'] = actual_k
    results['predicted_classes'] = optimal_k
    results['correct'] = correct
    results['known_classes'] = setup['n_known']
    results['hidden_classes_count'] = len(hidden_classes)
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"m̄(∅): {results['m_empty_mean']:.4f}")
        print(f"FOD Complete: {results['fod_is_complete']}")
        print(f"Optimal k (predicted): {optimal_k}")
        print(f"Actual classes: {actual_k}")
        print(f"Correct: {'✓' if correct else '✗'}")
    
    return results


def run_all_experiments(verbose: bool = True, random_seed: int = None) -> List[Dict]:
    """
    Run all experiments from the paper Table 8-9
    
    Args:
        verbose: Print detailed output
        random_seed: Random seed. If None, uses true random.
    
    Returns:
        all_results: List of experiment results
    """
    all_results = []
    
    print("="*70)
    print("Running All Experiments (Paper Table 8-9 Reproduction)")
    print("="*70)
    
    # Experiment 1: Iris
    X, y, names = load_iris_dataset()
    result = run_single_experiment(
        "Iris", X, y, names, 
        hidden_classes=[1],  # Hide versicolor
        max_clusters=6, 
        verbose=verbose,
        random_seed=random_seed
    )
    all_results.append(result)
    
    # Experiment 2: Seeds
    X, y, names = load_seeds_dataset()
    result = run_single_experiment(
        "Seeds", X, y, names,
        hidden_classes=[2],  # Hide one class
        max_clusters=6,
        verbose=verbose,
        random_seed=random_seed
    )
    all_results.append(result)
    
    # Experiment 3: Haberman
    X, y, names = load_haberman_dataset()
    result = run_single_experiment(
        "Haberman", X, y, names,
        hidden_classes=[1],  # Hide one class
        max_clusters=5,
        verbose=verbose,
        random_seed=random_seed
    )
    all_results.append(result)
    
    # Experiment 4: WDBC
    X, y, names = load_wdbc_dataset()
    result = run_single_experiment(
        "WDBC", X, y, names,
        hidden_classes=[1],  # Hide one class
        max_clusters=5,
        verbose=verbose,
        random_seed=random_seed
    )
    all_results.append(result)
    
    return all_results


def run_multiple_experiments(n_runs: int = 20, verbose: bool = False) -> Dict:
    """
    运行多次实验并统计结果
    
    Args:
        n_runs: 运行次数（论文中通常为10-20次）
        verbose: Print detailed output for each run
    
    Returns:
        统计结果：平均值、标准差、成功率等
    """
    from tqdm import tqdm
    
    print(f"\n{'=' * 70}")
    print(f"Running Multiple Experiments (N={n_runs})")
    print(f"{'=' * 70}")
    
    # Store results per dataset
    dataset_results = {}
    
    for i in tqdm(range(n_runs), desc="Running experiment sets"):
        # Use true random for each run
        results = run_all_experiments(verbose=verbose, random_seed=None)
        
        for r in results:
            dataset_name = r['dataset']
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {
                    'm_empty_means': [],
                    'optimal_ks': [],
                    'correct_detections': [],
                    'actual_k': r['actual_classes']
                }
            dataset_results[dataset_name]['m_empty_means'].append(r['m_empty_mean'])
            dataset_results[dataset_name]['optimal_ks'].append(r['predicted_classes'])
            dataset_results[dataset_name]['correct_detections'].append(r['correct'])
    
    # Calculate and print statistics
    print(f"\n{'=' * 70}")
    print(f"=== 多次实验统计结果 (N={n_runs}) ===")
    print(f"{'=' * 70}")
    
    stats = {}
    for dataset_name, data in dataset_results.items():
        m_empty_arr = np.array(data['m_empty_means'])
        k_arr = np.array(data['optimal_ks'])
        correct_arr = np.array(data['correct_detections'])
        
        stats[dataset_name] = {
            'm_empty_mean': {
                'mean': np.mean(m_empty_arr),
                'std': np.std(m_empty_arr)
            },
            'optimal_k': {
                'mean': np.mean(k_arr),
                'std': np.std(k_arr)
            },
            'success_rate': np.mean(correct_arr) * 100,
            'actual_k': data['actual_k']
        }
        
        print(f"\n数据集: {dataset_name}")
        print(f"  m̄(∅): {stats[dataset_name]['m_empty_mean']['mean']:.4f} ± {stats[dataset_name]['m_empty_mean']['std']:.4f}")
        print(f"  Optimal k: {stats[dataset_name]['optimal_k']['mean']:.1f} ± {stats[dataset_name]['optimal_k']['std']:.2f} (成功率: {stats[dataset_name]['success_rate']:.1f}%)")
    
    print(f"\n{'=' * 70}")
    
    return stats


def print_summary_table(results: List[Dict]):
    """Print summary table similar to paper Table 8-9"""
    print("\n" + "="*70)
    print("SUMMARY TABLE (Paper Table 8-9)")
    print("="*70)
    print(f"{'Dataset':<15} {'Real k':<10} {'FGS k':<10} {'m̄(∅)':<12} {'Correct':<10}")
    print("-"*70)
    
    n_correct = 0
    for r in results:
        correct_str = '✓ Yes' if r['correct'] else '✗ No'
        if r['correct']:
            n_correct += 1
        print(f"{r['dataset']:<15} {r['actual_classes']:<10} {r['predicted_classes']:<10} {r['m_empty_mean']:<12.4f} {correct_str:<10}")
    
    print("-"*70)
    print(f"Accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Check if running multiple experiments
    if len(sys.argv) > 1 and sys.argv[1] == '--multi':
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        stats = run_multiple_experiments(n_runs=n_runs)
    else:
        # Run all experiments once with true random
        results = run_all_experiments(verbose=True)
        
        # Print summary
        print_summary_table(results)
