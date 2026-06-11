import struct

class BinaryConverter:
    # --- HELPERS ---
    def _int_to_bin_str(self, n: int) -> str:
        """Manually converts integer to binary string."""
        if n == 0: return "0"
        binary: list[str] = []
        while n > 0:
            # Append remainder (0 or 1)
            binary.append(str(n % 2))
            n //= 2
        # Reverse list to get correct order
        return "".join(reversed(binary))

    def _pad_left(self, binary_str: str, width: int) -> str:
        """Manually adds leading zeros."""
        zeros_needed = width - len(binary_str)
        if zeros_needed > 0:
            return ("0" * zeros_needed) + binary_str
        return binary_str

    # --- INTEGER (64-bit Two's Complement) ---
    def convert_integer(self, decimal: int, bits: int = 64) -> str:
        # Validate input range for two's complement representation
        min_value = -(1 << (bits - 1)) 
        max_value = (1 << (bits - 1)) - 1 
        
        if decimal < min_value or decimal > max_value:
            raise ValueError(f"Value {decimal} cannot be represented in {bits} bits. Valid range: {min_value} to {max_value}.")
        
        # Handle negatives by shifting to unsigned equivalent
        if decimal < 0:
            decimal = (1 << bits) + decimal

        # Convert to binary (no masking needed since we validated the range)
        raw_binary = self._int_to_bin_str(decimal)
        
        return self._pad_left(raw_binary, bits)

    # --- FLOAT (IEEE 754 Double Precision 64-bit) ---
    def _float_fraction_to_bin(self, number: float, limit: int = 1074) -> str:
        """Converts fractional part (e.g., 0.625) to binary."""
        binary: list[str] = []
        while number > 0 and len(binary) < limit:
            number *= 2
            bit = int(number)
            binary.append(str(bit))
            number -= bit
        return "".join(binary)

    def convert_float_double(self, num: float) -> str:
        # 1. Edge Case: Zero
        if num == 0: return "0" * 64

        # 2. Sign Bit
        sign_bit = '1' if num < 0 else '0'
        num = abs(num)

        # 3. Split Integer and Fraction
        int_part = int(num)
        frac_part = num - int_part

        # 4. Convert parts (Using manual helper instead of format)
        int_str = self._int_to_bin_str(int_part)
        frac_str = self._float_fraction_to_bin(frac_part)

        # 5. Normalize
        exponent_unbiased = 0
        mantissa_str = ""

        if int_part > 0:
            # E.g., 101.1 -> 1.011 * 2^2
            exponent_unbiased = len(int_str) - 1  # = 2
            mantissa_str = int_str[1:] + frac_str
        else:
            # E.g., 0.00101 -> 1.01 * 2^-3
            first_one = frac_str.find('1')
            if first_one == -1: return "0" * 64 
            exponent_unbiased = -(first_one + 1)
            mantissa_str = frac_str[first_one + 1:]

        # 6. Exponent (Bias 1023, 11 bits)
        exponent_biased = exponent_unbiased + 1023  # -1024 to 1023
        # Manual conversion and padding
        exponent_raw = self._int_to_bin_str(exponent_biased)
        exponent_bits = self._pad_left(exponent_raw, 11)

        # 7. Mantissa (52 bits)
        if len(mantissa_str) < 52:
            mantissa_bits = mantissa_str + ("0" * (52 - len(mantissa_str)))
        else:
            mantissa_bits = mantissa_str[:52]

        return sign_bit + exponent_bits + mantissa_bits

    # --- DISPATCHER ---
    def convert(self, val: int | float) -> str:
        if isinstance(val, float): return self.convert_float_double(val)
        if isinstance(val, int): return self.convert_integer(val)
        raise ValueError("Invalid type")

    def revert(self, binary: str, dtype: type) -> int | float:
        """
        Reinterprets a 64-bit binary string as a specific type.
        dtype: int (for signed integer) or float (for double precision).
        """
        if len(binary) != 64:
            raise ValueError("Binary string must be exactly 64 bits.")
        
        # 1. Convert binary string to a Python integer (base 2)
        raw_int = int(binary, 2)
        
        # 2. Pack this integer as an UNSIGNED 64-bit integer ('Q')
        # This gives us the raw bit pattern in bytes, exactly as it is in the string.
        raw_bytes = struct.pack('>Q', raw_int)
        
        # 3. Unpack the raw bytes as the requested target type
        if dtype is int:
            # 'q' = interpret bytes as signed long long (Two's Complement)
            return struct.unpack('>q', raw_bytes)[0]
        elif dtype is float:
            # 'd' = interpret bytes as double (IEEE 754)
            return struct.unpack('>d', raw_bytes)[0]
        else:
            raise ValueError("dtype must be int or float")


if __name__ == "__main__":
    converter = BinaryConverter()
    
    example_1 = converter.convert(10)
    example_1_reverted = converter.revert(example_1, int)

    example_2 = converter.convert(10.625)
    example_2_reverted = converter.revert(example_2, float)

    example_3 = converter.convert(-10.625)
    example_3_reverted = converter.revert(example_3, float)

    example_4 = converter.convert(0.1)
    example_4_reverted = converter.revert(example_4, float)

    example_5 = converter.convert(-0.1)
    example_5_reverted = converter.revert(example_5, float)

    print("Binary (10):", example_1)
    print("Decimal (10):", example_1_reverted)

    print("Binary (10.625):", example_2)
    print("Decimal (10.625):", example_2_reverted)

    print("Binary (-10.625):", example_3)
    print("Decimal (-10.625):", example_3_reverted)

    print("Binary (0.1):", example_4)
    print("Decimal (0.1):", example_4_reverted)

    print("Binary (-0.1):", example_5)
    print("Decimal (-0.1):", example_5_reverted)
