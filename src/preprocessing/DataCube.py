class DataCube:

    def __init__(self, expression_mtx, location_mtx, relation=None):
        self.id = None
        self.expression = expression_mtx
        self.location = location_mtx
        self.relation = relation
        self.gene_names = None
        self.n = self.expression.shape[0]
        self.neighbor_graph = None

    def get_expression(self):
        """
            Return expression
        """
        return self.expression

    def get_location(self):
        """
            Return location
        """
        return self.location

    def get_size(self):
        """
            Return total cell count
        """
        return self.n

    def get_neighbor_graph(self):
        """
            Return neighbor graph
        """
        if self.neighbor_graph is None:
            raise Exception("Graph is not set. Currently neighbor graph is None")
        return self.neighbor_graph
