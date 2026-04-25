import unittest
import numpy as np
from features import calculate_angle, extract_features, get_feedback
from config import N_FEATURES
from unittest.mock import Mock

class TestFeatures(unittest.TestCase):
    
    def test_calculate_angle_90_degrees(self):
        """Test that a right angle returns 90 degrees"""
        a = [0, 0]
        b = [0, 1]
        c = [1, 1]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)
    
    def test_calculate_angle_180_degrees(self):
        """Test that a straight line returns 180 degrees"""
        a = [0, 0]
        b = [1, 0]
        c = [2, 0]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=1)
    
    def test_calculate_angle_0_degrees(self):
        """Test that overlapping points return 0 degrees"""
        a = [0, 1]
        b = [0, 0]
        c = [0, 1]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 0.0, places=1)
    
    def test_extract_features_length(self):
        """Test that extract_features returns exactly 135 features"""
        # Create mock landmarks
        landmarks = []
        for i in range(33):
            mock_lm = Mock()
            mock_lm.x = 0.5
            mock_lm.y = 0.5
            mock_lm.z = 0.0
            mock_lm.visibility = 1.0
            landmarks.append(mock_lm)
        
        features = extract_features(landmarks)
        self.assertEqual(len(features), N_FEATURES)
        self.assertEqual(len(features), 135)
    
    def test_get_feedback_bent_elbows(self):
        """Test feedback when elbows are too bent"""
        angles = {
            "left_elbow_angle": 140,
            "right_elbow_angle": 145,
            "left_shoulder_angle": 80,
            "right_shoulder_angle": 82
        }
        
        mock_config = Mock()
        mock_config.ELBOW_TOO_BENT = 150
        mock_config.ARM_TOO_LOW = 60
        mock_config.ARM_TOO_HIGH = 100
        mock_config.SYMMETRY_MAX = 20
        
        feedback = get_feedback(angles, mock_config)
        self.assertTrue(any("Straighten elbows" in msg for msg in feedback))
    
    def test_get_feedback_arms_too_low(self):
        """Test feedback when arms are not raised enough"""
        angles = {
            "left_elbow_angle": 170,
            "right_elbow_angle": 175,
            "left_shoulder_angle": 50,
            "right_shoulder_angle": 52
        }
        
        mock_config = Mock()
        mock_config.ELBOW_TOO_BENT = 150
        mock_config.ARM_TOO_LOW = 60
        mock_config.ARM_TOO_HIGH = 100
        mock_config.SYMMETRY_MAX = 20
        
        feedback = get_feedback(angles, mock_config)
        self.assertTrue(any("Raise arms higher" in msg for msg in feedback))
    
    def test_get_feedback_asymmetric(self):
        """Test feedback when arms are uneven"""
        angles = {
            "left_elbow_angle": 170,
            "right_elbow_angle": 175,
            "left_shoulder_angle": 60,
            "right_shoulder_angle": 85
        }
        
        mock_config = Mock()
        mock_config.ELBOW_TOO_BENT = 150
        mock_config.ARM_TOO_LOW = 60
        mock_config.ARM_TOO_HIGH = 100
        mock_config.SYMMETRY_MAX = 20
        
        feedback = get_feedback(angles, mock_config)
        self.assertTrue(any("Keep arms even" in msg for msg in feedback))
    
    def test_get_feedback_good_form(self):
        """Test that good form returns no feedback"""
        angles = {
            "left_elbow_angle": 176,
            "right_elbow_angle": 178,
            "left_shoulder_angle": 78,
            "right_shoulder_angle": 80
        }
        
        mock_config = Mock()
        mock_config.ELBOW_TOO_BENT = 150
        mock_config.ARM_TOO_LOW = 60
        mock_config.ARM_TOO_HIGH = 100
        mock_config.SYMMETRY_MAX = 20
        
        feedback = get_feedback(angles, mock_config)
        self.assertEqual(len(feedback), 0)

if __name__ == '__main__':
    unittest.main()
