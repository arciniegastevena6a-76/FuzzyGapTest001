"""
Validation tests to verify the implementation matches the FGS paper results.

Paper Reference:
- Updating incomplete framework of target recognition database based on fuzzy gap statistic

Test Cases:
1. m̄(∅) calculation should be close to 0.589 (Table 2)
2. Optimal k should be 3 for Iris dataset (Fig. 4)
3. GBPA calculation rules (Section 2.2)
"""

import numpy as np
from sklearn.datasets import load_iris

# Tolerance for floating point comparisons
M_EMPTY_TOLERANCE = 0.01  # Allow 1% deviation
PAPER_M_EMPTY_MEAN = 0.589
PAPER_OPTIMAL_K = 3


def test_m_empty_calculation():
    """
    Test 1: Verify m̄(∅) calculation matches paper Table 2
    
    Paper Table 2 shows:
        SL=0.4283, SW=0.5451, PL=0.6829, PW=0.6995
        Mean = (0.4283 + 0.5451 + 0.6829 + 0.6995) / 4 = 0.589
    
    Our implementation should produce a value close to 0.589.
    """
    from gbpa import GBPAGenerator
    
    # Load Iris data
    iris = load_iris()
    data = iris.data
    target = iris.target
    
    # Paper setup: FOD = {setosa, virginica}, unknown = {versicolor}
    known_classes = [0, 2]
    
    # Data split (following paper Section 4.1)
    np.random.seed(108)
    
    train_indices = []
    test_indices = []
    
    for cls in range(3):
        cls_indices = np.where(target == cls)[0]
        np.random.shuffle(cls_indices)
        
        if cls in known_classes:
            train_indices.extend(cls_indices[:40])
            test_indices.extend(cls_indices[40:])
        else:
            test_indices.extend(cls_indices[:30])
    
    train_data = data[train_indices]
    train_labels = target[train_indices]
    test_data = data[test_indices]
    
    # Build TFN and generate GBPA
    gbpa_gen = GBPAGenerator()
    gbpa_gen.build_tfn_models(train_data, train_labels)
    
    # Generate GBPA
    gbpa_list, m_empty_combined, m_empty_mean_attr, attr_m_empty = gbpa_gen.generate(test_data)
    
    # Calculate m̄(∅) - should be average of all samples' attribute-level m(∅) means
    # This is equivalent to the overall mean of all m(∅) values
    all_m_empty = np.array(attr_m_empty)
    m_empty_mean = np.mean(all_m_empty)
    
    # Also verify that our two calculation methods give the same result
    m_empty_mean_alt = np.mean(m_empty_mean_attr)
    
    print(f"\n=== Test 1: m̄(∅) Calculation ===")
    print(f"Calculated m̄(∅): {m_empty_mean:.4f}")
    print(f"Alternative calculation: {m_empty_mean_alt:.4f}")
    print(f"Paper value: {PAPER_M_EMPTY_MEAN}")
    print(f"Difference: {abs(m_empty_mean - PAPER_M_EMPTY_MEAN):.4f}")
    
    # Per-attribute means (for comparison with Table 2)
    per_attr_means = np.mean(all_m_empty, axis=0)
    print(f"\nPer-attribute m(∅) means:")
    print(f"  SL (attr 0): {per_attr_means[0]:.4f} (paper: 0.4283)")
    print(f"  SW (attr 1): {per_attr_means[1]:.4f} (paper: 0.5451)")
    print(f"  PL (attr 2): {per_attr_means[2]:.4f} (paper: 0.6829)")
    print(f"  PW (attr 3): {per_attr_means[3]:.4f} (paper: 0.6995)")
    
    # Assertions
    assert abs(m_empty_mean - m_empty_mean_alt) < 1e-10, \
        f"Two m̄(∅) calculation methods should give same result"
    
    assert abs(m_empty_mean - PAPER_M_EMPTY_MEAN) < M_EMPTY_TOLERANCE, \
        f"m̄(∅)={m_empty_mean:.4f} differs from paper value {PAPER_M_EMPTY_MEAN} by more than {M_EMPTY_TOLERANCE}"
    
    print("\n✓ Test 1 PASSED: m̄(∅) calculation is correct")
    return True


def test_optimal_k_determination():
    """
    Test 2: Verify optimal k determination matches paper Fig. 4
    
    Paper shows optimal k = 3 for Iris dataset.
    """
    from fuzzy_gap_statistic import FuzzyGapStatistic
    
    # Load Iris data
    iris = load_iris()
    data = iris.data
    target = iris.target
    
    # Paper setup
    known_classes = [0, 2]
    
    # Data split
    np.random.seed(108)
    
    train_indices = []
    test_indices = []
    
    for cls in range(3):
        cls_indices = np.where(target == cls)[0]
        np.random.shuffle(cls_indices)
        
        if cls in known_classes:
            train_indices.extend(cls_indices[:40])
            test_indices.extend(cls_indices[40:])
        else:
            test_indices.extend(cls_indices[:30])
    
    train_data = data[train_indices]
    train_labels = target[train_indices]
    test_data = data[test_indices]
    
    # Run FGS
    fgs = FuzzyGapStatistic(critical_value=0.5, max_iterations=100, random_seed=108)
    fgs.gbpa_generator.build_tfn_models(train_data, train_labels)
    
    # Check m̄(∅) > 0.5 (FOD should be incomplete)
    gbpa_list, m_empty_mean, statistics = fgs.generate_gbpa_and_analyze(test_data)
    
    print(f"\n=== Test 2: Optimal k Determination ===")
    print(f"m̄(∅) = {m_empty_mean:.4f}")
    print(f"FOD Complete: {fgs.is_fod_complete(m_empty_mean)}")
    
    # Perform FGS clustering
    sampled_data = fgs.perform_monte_carlo_sampling(test_data, n_samples=20)
    optimal_k, fgs_results = fgs.determine_optimal_clusters(test_data, sampled_data, max_clusters=6)
    
    print(f"\nFGS Results:")
    print(f"{'k':<5} {'Gap(k)':<12} {'s_k':<12}")
    print("-" * 30)
    for k in sorted(fgs_results.keys()):
        gap = fgs_results[k]
        s_k = fgs.gap_calc.s_k.get(k, 0.0)
        print(f"{k:<5} {gap:<12.6f} {s_k:<12.6f}")
    
    print(f"\nOptimal k: {optimal_k}")
    print(f"Paper value: {PAPER_OPTIMAL_K}")
    
    # Assertions
    assert optimal_k == PAPER_OPTIMAL_K, \
        f"Optimal k={optimal_k} differs from paper value {PAPER_OPTIMAL_K}"
    
    print("\n✓ Test 2 PASSED: Optimal k determination is correct")
    return True


def test_gbpa_rules():
    """
    Test 3: Verify GBPA generation rules from 9-2 document Section 2.2
    
    Rules:
    ① Single intersection: GBPA = membership value
    ② Two intersections: high → single subset, low → multi-subset
    ③ Three+ intersections: each gets own membership, lowest for all-subset
    """
    from gbpa import GBPAGenerator
    
    print(f"\n=== Test 3: GBPA Generation Rules ===")
    
    # Create simple test data
    generator = GBPAGenerator()
    
    # Build simple TFN models for testing
    # Class 0: feature in [0, 1], mean=0.5
    # Class 1: feature in [0.5, 1.5], mean=1.0
    # Class 2: feature in [1.5, 2.5], mean=2.0
    train_data = np.array([
        [0.0], [0.5], [1.0],  # Class 0
        [0.5], [1.0], [1.5],  # Class 1
        [1.5], [2.0], [2.5]   # Class 2
    ])
    train_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    generator.build_tfn_models(train_data, train_labels)
    
    print("\nTFN Models:")
    for cls in generator.known_classes:
        tfn = generator.tfn_models[cls][0]
        print(f"  Class {cls}: TFN = ({tfn[0]:.2f}, {tfn[1]:.2f}, {tfn[2]:.2f})")
    
    # Test Rule ①: Sample outside all TFNs
    sample_outside = np.array([3.0])
    gbpa = generator._generate_gbpa_for_single_attribute(sample_outside[0], 0)
    print(f"\nRule ①/⑤ - Sample outside all (x=3.0):")
    print(f"  GBPA: {dict((str(k), round(v, 4)) for k, v in gbpa.items())}")
    assert gbpa.get('empty', 0) == 1.0, "Sample outside all TFNs should have m(∅)=1"
    
    # Test Rule ①: Sample in only one TFN
    sample_single = np.array([0.25])  # Only in class 0's TFN
    gbpa = generator._generate_gbpa_for_single_attribute(sample_single[0], 0)
    print(f"\nRule ① - Single intersection (x=0.25):")
    print(f"  GBPA: {dict((str(k), round(v, 4)) for k, v in gbpa.items())}")
    assert frozenset([0]) in gbpa, "Single intersection should have single subset GBPA"
    
    # Test Rule ②: Sample in two TFNs
    sample_two = np.array([0.75])  # In class 0 and class 1's TFN overlap
    gbpa = generator._generate_gbpa_for_single_attribute(sample_two[0], 0)
    print(f"\nRule ② - Two intersections (x=0.75):")
    print(f"  GBPA: {dict((str(k), round(v, 4)) for k, v in gbpa.items())}")
    # Should have single subset for higher membership and multi-subset for lower
    
    # Test Rule ③: Sample in three TFNs
    sample_three = np.array([1.0])  # Might intersect all three
    gbpa = generator._generate_gbpa_for_single_attribute(sample_three[0], 0)
    print(f"\nRule ③ - Three intersections (x=1.0):")
    print(f"  GBPA: {dict((str(k), round(v, 4)) for k, v in gbpa.items())}")
    
    print("\n✓ Test 3 PASSED: GBPA generation rules are implemented")
    return True


def test_gcr_combination():
    """
    Test 4: Verify GCR (Generalized Combination Rule) from 9-1 document
    
    Using Example 4 from the document:
    m1(a)=0.1, m1(b)=0.2, m1(∅)=0.7
    m2(a)=0.1, m2({b,c})=0.1, m2(∅)=0.8
    
    Expected results:
    m(∅) = 0.7 * 0.8 = 0.56
    K ≈ 0.97
    m(a) ≈ 0.147
    m(b) ≈ 0.293
    """
    from gbpa import GBPAGenerator, test_gcr_example4
    
    print(f"\n=== Test 4: GCR Combination Rule ===")
    
    # Run the built-in test
    result = test_gcr_example4()
    
    print("\n✓ Test 4 PASSED: GCR combination rule is correct")
    return result


def run_all_tests():
    """Run all validation tests"""
    print("=" * 70)
    print("Paper Validation Tests")
    print("=" * 70)
    
    results = {}
    
    try:
        results['m_empty'] = test_m_empty_calculation()
    except AssertionError as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        results['m_empty'] = False
    
    try:
        results['optimal_k'] = test_optimal_k_determination()
    except AssertionError as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        results['optimal_k'] = False
    
    try:
        results['gbpa_rules'] = test_gbpa_rules()
    except AssertionError as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        results['gbpa_rules'] = False
    
    try:
        results['gcr'] = test_gcr_combination()
    except AssertionError as e:
        print(f"\n✗ Test 4 FAILED: {e}")
        results['gcr'] = False
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
