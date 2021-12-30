class Node:
    def __init__(self, ID , node_list = []):
        self.__id = ID
        self.__connections = node_list.copy()

    def get_id(self):
        return self.__id

    def neighbours(self):
        return self.__connections

    def get_degree(self):
        return len(self.__connections)

    def create_connection(self, node):

        # if node not in self.__connections:

        if (node not in self.__connections) and (node is not self):
            self.__connections.append(node)
            node.create_connection(self)


    def destroy_connection(self, id):
        
        for elm in self.__connections:
            if elm.get_id() == id:
                
                self.__connections.remove(elm)
                
                return True

        return False

    def __repr__(self):
        return self.__id








a = Node('1' ,)
b = Node('2' ,)
c = Node('3' ,)


a.create_connection(b)
b.create_connection(c)
c.create_connection(a)


print(a.neighbours())
print(b.neighbours())
print(c.neighbours())

