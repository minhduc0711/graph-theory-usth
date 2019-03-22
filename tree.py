class Tree:
    def __init__(self, val, parent=None):
        self.val = val
        self.children = []
        self.set_parent(parent)

    def set_parent(self, parent):
        self.parent = parent
        if parent:
            self.parent.children.append(self)

    def __str__(self):
        return self.val


def preorder(T):
    print(T.val, end="; ")
    for child in T.children:
        preorder(child)


def postorder(T):
    for child in T.children:
        postorder(child)
    print(T.val, end="; ")
