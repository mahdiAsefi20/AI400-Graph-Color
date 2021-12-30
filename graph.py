COLOR = ['RED' , 'BLUE' , 'WHITE' , 'YELLOW' , 'GREEN']


class Node:
    def __init__(self, ID , node_list = [] , color = None):
        self.__id = ID
        self.__connections = node_list.copy()
        self.__color = color

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

    def colorize(self , color):
        self.__color = color

    def get_color(self):
        return self.__color

    def __repr__(self):
        return  str(self.__id)



class Graph:

    def __init__(self, name = 'default' , node_list = []):
        self.nodes = node_list
        self.name = name


    def is_coloring_valid(self):

        for node in self.nodes:

            my_color = node.get_color()

            for neighbour in node.neighbours():
                
                if my_color == neighbour.get_color():
                    return False

        return True

    
    def score(self): # Fitness Function
        # Needed for genetics Algorithm
        pass

    @staticmethod
    def read_file(pth):
        with open(pth , 'r') as file:
            num_of_nodes = int( file.readline() )
            
            # node_list = [Node(name) for name in range(1 , num_of_nodes + 1)]    # list comperhension
            
            node_list = { str(name): Node(name) for name in range(1 , num_of_nodes + 1)}

            for index in range(1 , num_of_nodes + 1 ):
                file_line = file.readline()
                current_node_list = file_line.split(' ')[:-1]
                
                for node_key in current_node_list:
                    if (node_key == '-1'):
                        break

                    node_list[str(index)].create_connection(
                        node_list[node_key]
                    )

        return Graph(node_list.values())


# my_graph = Graph.read_file('sample-graph.gp')



a = Node('1' ,)
b = Node('2' ,)
c = Node('3' ,)


a.create_connection(b)
b.create_connection(c)
c.create_connection(a)

a.colorize('RED')
b.colorize('RED')
c.colorize('GREEN')

my_graph = Graph(node_list= [a,b,c])
print(
    my_graph.is_coloring_valid()
)



###### For Writing Custom File ######
## Line 0: Number Of Node
## Line 1 - NON: Connections to other node
### Reminder. Note that at the end of each line from line 1 to line NON you must
### leave a blank space