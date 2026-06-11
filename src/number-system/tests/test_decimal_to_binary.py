import unittest
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import from the file with hyphen in name
import importlib.util
spec = importlib.util.spec_from_file_location("decimal_to_binary", "decimal-to-binary.py")
decimal_to_binary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decimal_to_binary_module)
DecimalToBinary = decimal_to_binary_module.DecimalToBinary


class TestDecimalToBinary(unittest.TestCase):
    """Test cases for DecimalToBinary class."""
    
    def setUp(self):
        self.converter = DecimalToBinary()
    
    def test_convert_integer_positive(self):
        """Test positive integer conversion."""
        self.assertEqual(self.converter.convert_integer(10), "1010")
        self.assertEqual(self.converter.convert_integer(1), "1")
        self.assertEqual(self.converter.convert_integer(0), "0")
        self.assertEqual(self.converter.convert_integer(255), "11111111")
    
    def test_convert_integer_negative(self):
        """Test negative integer conversion."""
        self.assertEqual(self.converter.convert_integer(-10), "-1010")
        self.assertEqual(self.converter.convert_integer(-1), "-1")
        self.assertEqual(self.converter.convert_integer(-255), "-11111111")
    
    def test_convert_integer_twos_complement_positive(self):
        """Test positive integer two's complement conversion."""
        self.assertEqual(self.converter.convert_integer_twos_complement(10, 8), "00001010")
        self.assertEqual(self.converter.convert_integer_twos_complement(0, 8), "00000000")
        self.assertEqual(self.converter.convert_integer_twos_complement(127, 8), "01111111")
        self.assertEqual(self.converter.convert_integer_twos_complement(1, 4), "0001")
    
    def test_convert_integer_twos_complement_negative(self):
        """Test negative integer two's complement conversion."""
        self.assertEqual(self.converter.convert_integer_twos_complement(-1, 8), "11111111")
        self.assertEqual(self.converter.convert_integer_twos_complement(-10, 8), "11110110")
        self.assertEqual(self.converter.convert_integer_twos_complement(-128, 8), "10000000")
        self.assertEqual(self.converter.convert_integer_twos_complement(-1, 4), "1111")
    
    def test_convert_float_positive(self):
        """Test positive float conversion."""
        self.assertEqual(self.converter.convert_float(10.625), "1010.101")
        self.assertEqual(self.converter.convert_float(5.25), "101.01")
        self.assertEqual(self.converter.convert_float(0.5), "0.1")
        self.assertEqual(self.converter.convert_float(0.75), "0.11")
        self.assertEqual(self.converter.convert_float(1.0), "1")
    
    def test_convert_float_negative(self):
        """Test negative float conversion."""
        self.assertEqual(self.converter.convert_float(-10.625), "-1010.101")
        self.assertEqual(self.converter.convert_float(-5.25), "-101.01")
        self.assertEqual(self.converter.convert_float(-0.5), "-0.1")
    
    def test_convert_float_zero(self):
        """Test zero float conversion."""
        self.assertEqual(self.converter.convert_float(0.0), "0.0")
    
    def test_convert_float_precision(self):
        """Test float conversion with different precision."""
        # Test with limited precision
        result = self.converter.convert_float(0.1, precision=4)
        self.assertEqual(result, "0.0001")  # 0.1 in binary (approximation)
        
        # Test with high precision
        result = self.converter.convert_float(0.125, precision=10)
        self.assertEqual(result, "0.001")
    
    def test_convert_generic_integers(self):
        """Test generic convert method with integers."""
        self.assertEqual(self.converter.convert(10), "1010")
        self.assertEqual(self.converter.convert(-10), "-1010")
        self.assertEqual(self.converter.convert(0), "0")
    
    def test_convert_generic_floats(self):
        """Test generic convert method with floats."""
        self.assertEqual(self.converter.convert(10.625), "1010.101")
        self.assertEqual(self.converter.convert(-5.25), "-101.01")
        self.assertEqual(self.converter.convert(0.0), "0.0")
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Large numbers
        self.assertEqual(self.converter.convert_integer(1024), "10000000000")
        
        # Two's complement overflow protection
        result = self.converter.convert_integer_twos_complement(255, 8)
        self.assertEqual(result, "11111111")
        
        # Float with no fractional part
        self.assertEqual(self.converter.convert_float(8.0), "1000")
        
        # Very small fractional part
        result = self.converter.convert_float(0.0625)  # 1/16
        self.assertEqual(result, "0.0001")
    
    def test_bit_width_variations(self):
        """Test different bit widths for two's complement."""
        # 4-bit representations
        self.assertEqual(self.converter.convert_integer_twos_complement(7, 4), "0111")
        self.assertEqual(self.converter.convert_integer_twos_complement(-8, 4), "1000")
        
        # 16-bit representations
        self.assertEqual(self.converter.convert_integer_twos_complement(1, 16), "0000000000000001")
        self.assertEqual(self.converter.convert_integer_twos_complement(-1, 16), "1111111111111111")


if __name__ == '__main__':
    unittest.main()
