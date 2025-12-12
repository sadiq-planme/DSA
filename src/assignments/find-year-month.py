import os
import re
from dataclasses import dataclass


# Configuration
INPUT_DATA: str = "/media/sadiq-vali/sadakHDD/Problem - Find Year/Assignment/data"
START_YEAR: int = 1000
END_YEAR: int = 9333
MONTH_NAMES: list[str] = [
    "January.txt", "February.txt", "March.txt", "April.txt", "May.txt", "June.txt",
    "July.txt", "August.txt", "September.txt", "October.txt", "November.txt", "December.txt"
]


@dataclass
class YearMonthTuple:
    year: int
    month: str
    filepath: str


class BinarySearch:

    def __init__(self, 
        input_data: str = INPUT_DATA, 
        start_year: int = START_YEAR, 
        end_year: int = END_YEAR, 
        month_names: list[str] = MONTH_NAMES
    ):
        self.input_data: str = input_data
        self.start_year: int = start_year
        self.end_year: int = end_year
        self.month_names: list[str] = month_names
        self.year_month_tuples: list[YearMonthTuple] = []

        self.prepare_list_of_tuples_for_each_month_in_year()

    def prepare_list_of_tuples_for_each_month_in_year(self):
        for year in range(self.start_year, self.end_year + 1):
            for month in self.month_names:
                file_path = os.path.join(self.input_data, str(year), month)
                self.year_month_tuples.append(YearMonthTuple(year, month, file_path))

    def extract_value_from(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                match = re.search(r'\[\[Number of people:\s*(\d+)\]\]', content)
                if match:
                    return int(match.group(1))
        except Exception as e:
            print(f"Error extracting value from file: {filepath} - {e}")
            return None

    def search(self, target_value: int):
        low = 0
        high = len(self.year_month_tuples) - 1
        while low <= high:
            mid = (low + high) // 2
            filepath = self.year_month_tuples[mid].filepath
            current_value = self.extract_value_from(filepath)
            if current_value is None:
                print(f"Could not read data at filepath: {filepath}")
                return None
            if current_value == target_value:
                return self.year_month_tuples[mid]
            elif current_value < target_value:
                low = mid + 1
            else:
                high = mid - 1
        return None   


obj = BinarySearch()
target_value = 47359253
result = obj.search(target_value)

if result is not None:
    print(f"\n\nTarget value {target_value} found in:")
    print(f"Year: {result.year}")
    print(f"Month: {result.month}")
    print(f"Filepath: {result.filepath}\n\n")
else:
    print(f"\nTarget value {target_value} not found")
