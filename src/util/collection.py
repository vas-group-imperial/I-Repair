class Collection(object):
    def __init__(self, collection):

        self.collection_dict = dict()

        for tup in collection:
            for i, v in enumerate(tup):
                self.collection_dict[(i, v)] = tup

    # TODO: remove once we have real ranking
    def get_length(self):

        return len(self.collection_dict)

    def lookup_by_first_element(self, e):

        return self.collection_dict.get((0, e), None)

    def lookup_by_second_element(self, e):

        return self.collection_dict.get((1, e), None)
