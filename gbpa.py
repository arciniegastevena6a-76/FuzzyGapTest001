"""
Generalized Basic Probability Assignment (GBPA) generation module
严格按照邓勇的强约束GBPA生成方法和GCR组合规则
关键修正：区分属性级m(Φ)平均值和GCR组合后的m(Φ)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

# Numerical tolerance constants
GCR_COMPLETE_CONFLICT_TOLERANCE = 1e-10  # Tolerance for K=1 check in GCR (9-1 document formula 14)
MIN_GBPA_SUM = 1e-10  # Minimum sum to avoid division by zero
DEGENERATE_CLUSTER_THRESHOLD = 1e-6  # Threshold for detecting degenerate clusters


class GBPAGenerator:
    """
    GBPA Generator based on Triangular Fuzzy Numbers
    严格按照图8-11中的强约束方法和GCR组合规则
    """

    def __init__(self):
        """Initialize GBPA Generator"""
        self.tfn_models = {}  # {class_label: {feature_idx: (min, mean, max)}}
        self.known_classes = []

    def build_tfn_models(self, train_data: np.ndarray,
                        train_labels: np.ndarray) -> Dict:
        """
        Build Triangular Fuzzy Number models
        按照图3-4的方式构建TFN（对每个类别、每个属性）

        Args:
            train_data: Training data (n_samples, n_features) - 原始数据
            train_labels: Training labels (n_samples,)

        Returns:
            tfn_models: Dictionary of TFN models
        """
        self.known_classes = sorted(np.unique(train_labels))
        self.tfn_models = {}

        n_features = train_data.shape[1]

        for cls in self.known_classes:
            cls_mask = train_labels == cls
            cls_data = train_data[cls_mask]

            if len(cls_data) == 0:
                continue

            features_tfn = {}
            for j in range(n_features):
                feature_values = cls_data[:, j]
                min_val = np.min(feature_values)
                mean_val = np.mean(feature_values)
                max_val = np.max(feature_values)

                features_tfn[j] = (min_val, mean_val, max_val)

            self.tfn_models[cls] = features_tfn

        return self.tfn_models

    def _triangular_membership(self, x: float, min_val: float,
                              mean_val: float, max_val: float) -> float:
        """
        Calculate triangular membership function
        图1的三角隶属函数 Eq.(5)
        """
        if x < min_val:
            return 0.0
        elif min_val <= x <= mean_val:
            if mean_val == min_val:
                return 1.0
            return (x - min_val) / (mean_val - min_val)
        elif mean_val < x <= max_val:
            if max_val == mean_val:
                return 1.0
            return (max_val - x) / (max_val - mean_val)
        else:
            return 0.0

    def _generate_gbpa_for_single_attribute(self, attribute_value: float,
                                           feature_idx: int) -> Dict[str, float]:
        """
        Generate GBPA for a single attribute
        严格按照图8-10中的强约束规则（2.2节）

        核心规则：
        ① 当样本与某命题的三角模糊数表示模型相交时，
           相交点的纵坐标即为该样本支持命题的GBPA
        ② 当样本与多个命题的三角模糊数表示模型相交时，
           纵坐标高点为该样本支持单子集命题的GBPA，
           纵坐标低点为该样本支持多子集命题的GBPA
        ③ 如果累积之和超过1，归一化且m(Φ)=0；
           如果累积之和未超过1，生成m(Φ)
        ⑤ m(Φ)的生成准则：当样本与所有命题表示的三角形模糊数
           没有相交时，m(Φ)=1
        """
        if not self.tfn_models:
            raise ValueError("TFN models not built.")

        # Step 1: 计算与每个类别TFN的隶属度（交点纵坐标）
        memberships = {}
        for cls in self.known_classes:
            if feature_idx not in self.tfn_models[cls]:
                memberships[cls] = 0.0
                continue

            min_val, mean_val, max_val = self.tfn_models[cls][feature_idx]
            membership = self._triangular_membership(
                attribute_value, min_val, mean_val, max_val
            )
            memberships[cls] = membership

        # Step 2: 找出所有有交点的类别
        active_classes = [cls for cls, mem in memberships.items() if mem > 0]

        gbpa = {}

        # 规则⑤：与所有命题都不相交
        if len(active_classes) == 0:
            gbpa['empty'] = 1.0
            return gbpa

        # Step 3: 按照强约束规则生成GBPA
        # 按隶属度降序排列
        sorted_classes = sorted(active_classes,
                              key=lambda c: memberships[c],
                              reverse=True)

        # 规则①②：生成单子集和多子集命题
        if len(active_classes) == 1:
            # 只与一个类别相交 - 规则①
            cls = active_classes[0]
            gbpa[frozenset([cls])] = memberships[cls]

        elif len(sorted_classes) == 2:
            # 与两个类别相交 - 规则②
            cls1, cls2 = sorted_classes[0], sorted_classes[1]

            # 高点 → 单子集
            gbpa[frozenset([cls1])] = memberships[cls1]

            # 低点 → 二元多子集
            multi_subset = frozenset([cls1, cls2])
            gbpa[multi_subset] = memberships[cls2]

        elif len(sorted_classes) >= 3:
            # 与三个或更多类别相交 - 规则②③
            # According to 9-2 document Section 2.2:
            # - Highest ordinate → single subset proposition GBPA
            # - Second highest → binary multi-subset proposition GBPA
            # - Third highest → ternary multi-subset proposition GBPA
            # - Continue for more intersections
            
            # 最高点 → 单子集
            gbpa[frozenset([sorted_classes[0]])] = memberships[sorted_classes[0]]
            
            # 处理所有相交的类别，生成累积的多子集命题
            # Process all intersecting classes to generate cumulative multi-subset propositions
            for i in range(1, len(sorted_classes)):
                # 第i+1高点 → (i+1)元多子集命题
                # The (i+1)th highest point → (i+1)-element multi-subset proposition
                multi_subset = frozenset(sorted_classes[:i+1])
                gbpa[multi_subset] = memberships[sorted_classes[i]]

        # Step 4: 规则④ - 归一化或生成m(Φ)
        total_support = sum(gbpa.values())

        if total_support > 1.0:
            # 累积之和超过1，归一化
            for key in list(gbpa.keys()):
                gbpa[key] = gbpa[key] / total_support
            gbpa['empty'] = 0.0
        else:
            # 累积之和未超过1，剩余分配给空集
            gbpa['empty'] = 1.0 - total_support

        return gbpa

    def _generalized_combination_rule(self, m1: Dict, m2: Dict) -> Dict:
        """
        GCR - 广义组合规则
        严格按照图11中的公式(2)-(5)

        关键公式：
        m(A) = (1-m(Φ)) * Σ m₁(B)m₂(C) / (1-K)  ... (2)
        K = Σ m₁(B)m₂(C), B∩C=Φ                 ... (3)
        m(Φ) = m₁(Φ) * m₂(Φ)                     ... (4)
        m(Φ) = 1  当且仅当 K = 1                 ... (5)

        适用条件（图12、图17）：
        - 辨识框架不完整时适用
        - 辨识框架完整但干扰较小时适用
        - 辨识框架完整但干扰较大时不适用
        """
        combined = {}

        # 提取 m(Φ)
        m1_empty = m1.get('empty', 0.0)
        m2_empty = m2.get('empty', 0.0)

        # 计算 m(Φ) [公式(4)]
        m_combined_empty = m1_empty * m2_empty

        # 计算冲突系数 K [公式(3)]
        K = 0.0
        for B in m1:
            if B == 'empty':
                continue
            for C in m2:
                if C == 'empty':
                    continue
                if len(B & C) == 0:  # B∩C = Φ
                    K += m1[B] * m2[C]

        # 完全冲突 [公式(5)] - 9-1 document formula (14)
        # m(Φ) = 1 if and only if K = 1
        # Use stricter tolerance for numerical precision
        if K >= 1.0 - GCR_COMPLETE_CONFLICT_TOLERANCE:
            combined['empty'] = 1.0
            return combined

        # 组合非空集 [公式(2)]
        # 关键：(1-m(Φ)) = 1 - m₁(Φ) * m₂(Φ)
        factor = (1 - m_combined_empty) / (1 - K)

        for B in m1:
            if B == 'empty':
                continue
            for C in m2:
                if C == 'empty':
                    continue

                intersection = B & C

                if len(intersection) > 0:
                    if intersection not in combined:
                        combined[intersection] = 0.0
                    combined[intersection] += factor * m1[B] * m2[C]

        # 空集 [公式(4)]
        combined['empty'] = m_combined_empty

        return combined

    def generate_gbpa_for_sample(self, sample: np.ndarray) -> Tuple[Dict, float, float, List[float]]:
        """
        为单个样本生成GBPA

        关键修正：
        - m_empty_combined: GCR组合后的空集mass（用于后续分类）
        - m_empty_mean_attribute: 属性级空集mass的平均值（用于判断FOD完整性）

        流程（图17）：
        1. 对每个属性生成独立的GBPA（鉴别生成GBPA）
        2. 使用GCR逐个组合（GCR合成）

        Returns:
            combined_gbpa: 组合后的GBPA
            m_empty_combined: GCR组合后的样本级空集mass
            m_empty_mean_attribute: 属性级空集mass的平均值（用于判断FOD！）
            attribute_m_empty: 每个属性的空集mass列表
        """
        n_features = len(sample)

        # Step 1: 为每个属性生成GBPA（证据内部）
        attribute_gbpas = []
        attribute_m_empty = []

        for j in range(n_features):
            attr_gbpa = self._generate_gbpa_for_single_attribute(sample[j], j)
            attribute_gbpas.append(attr_gbpa)
            attribute_m_empty.append(attr_gbpa.get('empty', 0.0))

        # Step 2: 使用GCR逐个组合（证据间组合）
        combined_gbpa = attribute_gbpas[0]

        for i in range(1, len(attribute_gbpas)):
            combined_gbpa = self._generalized_combination_rule(
                combined_gbpa, attribute_gbpas[i]
            )

        # GCR组合后的 m(Φ)
        m_empty_combined = combined_gbpa.get('empty', 0.0)

        # 属性级 m(Φ) 的平均值（论文中用这个判断FOD完整性！）
        m_empty_mean_attribute = np.mean(attribute_m_empty)

        return combined_gbpa, m_empty_combined, m_empty_mean_attribute, attribute_m_empty

    def generate(self, data: np.ndarray,
                labels: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[List[float]]]:
        """
        Generate GBPA for all test samples

        修正：返回两种m(Φ)
        - m_empty_combined_array: GCR组合后的（用于后续分类）
        - m_empty_mean_attribute_array: 属性级平均（用于判断FOD完整性）

        Returns:
            gbpa_list: List of GBPA dictionaries for all samples
            m_empty_combined_array: GCR组合后的空集mass数组
            m_empty_mean_attribute_array: 属性级平均空集mass数组（用于判断FOD！）
            attribute_m_empty_list: List of attribute-level empty masses for each sample
        """
        if not self.tfn_models:
            raise ValueError("TFN models not built.")

        n_samples = data.shape[0]
        gbpa_list = []
        m_empty_combined_array = np.zeros(n_samples)
        m_empty_mean_attribute_array = np.zeros(n_samples)
        attribute_m_empty_list = []

        for i in range(n_samples):
            sample = data[i]
            gbpa, m_empty_combined, m_empty_mean_attr, attr_m_empty = \
                self.generate_gbpa_for_sample(sample)

            gbpa_list.append(gbpa)
            m_empty_combined_array[i] = m_empty_combined
            m_empty_mean_attribute_array[i] = m_empty_mean_attr
            attribute_m_empty_list.append(attr_m_empty)

        return gbpa_list, m_empty_combined_array, m_empty_mean_attribute_array, attribute_m_empty_list

    def calculate_mean_empty_mass(self, m_empty_array: np.ndarray) -> float:
        """
        计算平均空集mass（所有样本间）
        论文 Eq. (20)
        """
        return np.mean(m_empty_array)

    def analyze_empty_mass_statistics(self, m_empty_combined_array: np.ndarray,
                                     m_empty_mean_attribute_array: np.ndarray,
                                     attribute_m_empty_list: List[List[float]]) -> Dict:
        """
        分析空集mass的统计特征
        用于判断系统状态（图17的判断逻辑）

        修正：区分GCR组合后的m(Φ)和属性级平均m(Φ)

        Returns:
            statistics: 包含各级别统计信息的字典
        """
        # 属性级平均的统计（用于判断FOD完整性）
        attribute_mean_level_mean = np.mean(m_empty_mean_attribute_array)
        attribute_mean_level_std = np.std(m_empty_mean_attribute_array)

        # GCR组合后的统计（用于参考）
        combined_level_mean = np.mean(m_empty_combined_array)
        combined_level_std = np.std(m_empty_combined_array)

        # 属性内统计（证据内部）
        attribute_m_empty_array = np.array(attribute_m_empty_list)
        attribute_level_mean = np.mean(attribute_m_empty_array, axis=0)
        attribute_level_overall_mean = np.mean(attribute_m_empty_array)

        statistics = {
            'attribute_mean_level': {  # 用于判断FOD完整性
                'mean': attribute_mean_level_mean,
                'std': attribute_mean_level_std,
                'min': np.min(m_empty_mean_attribute_array),
                'max': np.max(m_empty_mean_attribute_array)
            },
            'combined_level': {  # GCR组合后（仅供参考）
                'mean': combined_level_mean,
                'std': combined_level_std,
                'min': np.min(m_empty_combined_array),
                'max': np.max(m_empty_combined_array)
            },
            'attribute_level': {  # 每个属性的统计
                'per_attribute_mean': attribute_level_mean,
                'overall_mean': attribute_level_overall_mean,
                'max_mean': np.max(attribute_level_mean),
                'min_mean': np.min(attribute_level_mean)
            }
        }

        return statistics