# A single node of a singly linked list
class Node:
  # constructor
  def __init__(self, data, next=None, prev=None, index=None): 
    self.data = data
    self.index = index
    self.next = next
    self.prev = prev

  def get_index(self):
     return self.index

  def get_next(self):
     return self.next
  
  def set_next(self, n):
     self.next = n

  def get_prev(self):
     return self.prev
  
  def set_prev(self, p):
     self.prev = p

  def print_data(self):
     print(self.data)

# A Doubly Linked List 
class DoubleyLinkedList:
    
    def __init__(self): # Initilized with no elements in the LL
       self.head = None
       self.tail = None
       self.size = 0

    def get_size(self):
       return self.size
    
    def push_head(self, d):
       self.size += 1
       new_node = Node(d)
       new_node.index = self.size
       new_node.set_next(self.head)

       if self.head != None: # if the list is not empty
          self.head.prev = new_node
          self.head = new_node
          new_node.set_prev(None)
       else: # if the list is empty
          self.head = new_node
          self.tail = new_node
          new_node.set_prev(None)

    def push_tail(self, d):
       self.size += 1
       new_node = Node(d)
       new_node.set_prev(self.tail)

       if self.tail != None: # if the list is not empty
          self.tail.next = new_node
          new_node.next = None
          self.tail = new_node
       else: # if the list is empty
          self.head = new_node
          self.tail = new_node
          new_node.set_next(None)
    
    def peak_head(self):
      if self.head == None: # checks whether list is empty or not
        print("List is empty")
      else:
        return self.head.data
      
    def peak_tail(self):
      if self.tail == None: # checks whether list is empty or not
        print("List is empty")
      else:
        return self.tail.data

    def pop_head(self):
       if self.head == None:
          print('The List is Empty :(')

       else:
          temp = self.head
          temp.next.prev = None
          self.head = temp.next
          temp.next = None
          self.size -= 1
          return temp.data

    def pop_tail(self):
       if self.head == None:
          print('The List is Empty :(')

       else:
          temp = self.tail
          temp.prev.next = None
          self.tail = temp.prev
          temp.prev = None
          self.size -= 1
          return temp.data
       
    def insert_after(self, temp_node, new_data):
      if temp_node == None:
        print("Given node is empty")
      else:
        new_node = Node(new_data)
        new_node.next = temp_node.next
        temp_node.next = new_node
        new_node.prev = temp_node
        if new_node.next != None:
          new_node.next.prev = new_node
        
        if temp_node == self.tail:
          self.tail = new_node 

      self.size += 1

    def insert_before(self, temp_node, new_data):
       if temp_node == None:
        print("Given node is empty")
       else:
        new_node = Node(new_data)
        new_node.prev = temp_node.prev
        temp_node.prev = new_node
        new_node.next = temp_node
        if new_node.prev != None:
          new_node.prev.next = new_node
        
        if temp_node == self.head: # checks whether new node is being added before the first element
          self.head = new_node # makes new node the new head

       self.size += 1

    def insert_at(self, n, d):
      if n < 0 or n > self.size:
         print("Invalid position: {}, for insertion".format(n))
         return
      
      if n == 0:
         self.push_head(d)
      elif n == self.size:
         self.push_tail(d)
      else:
         new_node = Node(d)
         current_node = self.head
         for i in range(n-1):
               current_node = current_node.get_next()
         
         new_node.set_next(current_node.get_next())
         new_node.set_prev(current_node)
         current_node.get_next().set_prev(new_node)
         current_node.set_next(new_node)
         self.size += 1

    def remove_at(self, n):
        if n < 0 or n >= self.size:
            print("Invalid position: {}, for removal".format(n))
            return
        
        if n == 0:
            return self.pop_head()
        elif n == self.size - 1:
            return self.pop_tail()
        else:
            current_node = self.head
            for i in range(n):
                current_node = current_node.get_next()
            
            data = current_node.data
            current_node.get_prev().set_next(current_node.get_next())
            current_node.get_next().set_prev(current_node.get_prev())
            current_node.set_next(None)
            current_node.set_prev(None)
            self.size -= 1
            return data

    def print(self, field=None):
       if self.head == None and self.tail == None:
          print('The List is Empty :(')
       else:
         current_node = self.head
       if field:
         for index in range(self.size):
            print("Node: {}".format(index))
            print(current_node.data[field])
            print()
            current_node = current_node.next
       else:
         for index in range(self.size):
            print("Node: {}".format(index))
            print(current_node.data)
            print()
            current_node = current_node.next

   #  def removeWords(self, text: str):
   #     removing_words = text.split()
   #     curr = self.head


