class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self  # The object itself acts as the iterator

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

# Usage
obj = MyIterator([1, 2, 3])
for item in obj:  # Implicitly calls __iter__ and then __next__
    print(item)
