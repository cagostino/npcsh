class Node:
       def __init__(self, value):
           self.value = value
           self.next = None

   class LinkedList:
       def __init__(self):
           self.head = None

       def append(self, value):
           new_node = Node(value)
           if not self.head:
               self.head = new_node
               return
           last = self.head
           while last.next:
               last = last.next
           last.next = new_node

   # Test for class Node
   print("This is a sample LinkedList implementation.")