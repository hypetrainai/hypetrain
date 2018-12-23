class InputLayer():
    def GetName(self):
        return "Input"

    def Validate(self, list_nodes):
        return True, None

    def GenerateInit(self, prev_node, node, code_init):
        return

    def GenerateForward(self, prev_node, node, code_forward):
        return
