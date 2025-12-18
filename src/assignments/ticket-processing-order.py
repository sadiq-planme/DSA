"""
BUsiness =>  
first class  =>
economy => 

NOTE : CLASS, DA, TN

everyone has ticket noum > incresing number

differently abled  first before other users of the same class
first > business > economy

lower ticket high priority > higher ticket number because they got it first FIFO

ticket num, class, 

case 1 =>


root



def comaprator_function(per1: Person, per2: Person) -> bool: # Here bool represents per1 has more priority than per2 or not 
    FIRST CLASS
        DA
        not DA
        lowest TN
    BUSINESS
        DA
        not DA
        lowest TN
    ECONOMY
        DA
        not DA
        lowest TN







differently abled > 

TN: 1
C: Business
DA: true
priority: 

TN: 2
C: first
DA: true
"""

from dataclasses import dataclass
from enum import Enum

class TicketClass(Enum):
    Business = "Business"
    FirstClass = "FirstClass"
    Economy = "Economy"


@dataclass
class Person:
    differenlty_abled: bool
    category: TicketClass
    ticketNum: int

ClassPriorityMap: dict[TicketClass, int] = {
    TicketClass.FirstClass: 3,
    TicketClass.Business: 2,
    TicketClass.Economy: 1
}

        
class MaxHeap:
    
    def __init__(self):
        self.heap = []
        
    def build_heap(self, persons: list[Person]):
        total = len(persons)
        if total == 0:
            return
        self.heap = persons[:]
        for internal_node_id in range((total - 2) >> 1, -1, -1):
            self._heapify_down(internal_node_id, total)
            
    def comparator_function(self, per1: Person, per2: Person) -> bool:
        """
        returns true if per1 has more priority than per2
        """
        per1_class_val = ClassPriorityMap[per1.category]
        per2_class_val = ClassPriorityMap[per2.category]
        
        if per1_class_val != per2_class_val:
            return per1_class_val > per2_class_val
        
        if per1.differenlty_abled != per2.differenlty_abled:
            return per1.differenlty_abled
        
        return per1.ticketNum < per2.ticketNum
            
    def _heapify_down(self, parent_id: int, total: int):
        if parent_id > ((total - 2) >> 1):
            return
        
        left_child_id = (parent_id << 1) + 1
        right_child_id = (parent_id << 1) + 2
        
        # assume parent is having heighest priority than the children
        high_priority = parent_id
        
        if left_child_id < total and self.comparator_function(self.heap[left_child_id], self.heap[high_priority]):
            high_priority = left_child_id
        if right_child_id < total and self.comparator_function(self.heap[right_child_id], self.heap[high_priority]):
            high_priority = right_child_id
        
        if high_priority == parent_id:
            return
        
        self.heap[high_priority], self.heap[parent_id] = self.heap[parent_id], self.heap[high_priority]
        self._heapify_down(high_priority, total)
        
    def pop(self) -> Person:
        if len(self.heap) == 0:
            return None
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self._heapify_down(0, len(self.heap) - 1)
        return self.heap.pop()

        
        
        
        
persons: list[Person] = [
        Person(False, TicketClass.Economy, 100),    
        Person(False, TicketClass.Business, 200),   
        Person(True,  TicketClass.Business, 205),   
        Person(False, TicketClass.FirstClass, 305), 
        Person(False, TicketClass.FirstClass, 301), 
        Person(True,  TicketClass.Economy, 102),    
    ]
    
heap = MaxHeap()
heap.build_heap(persons)

print()
print("The People Processing order")
print()

while True:
    person = heap.pop()
    if not person:
        break
    print(person)
    print()

print("All People processed successfully")
