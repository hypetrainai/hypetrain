from validator_layer import RequiredNodeValidator

class CELossLayer(RequiredNodeValidator):
    def GetName(self):
        return "CE Loss"

    def Validate(self, list_nodes):
        return RequiredNodeValidator.Validate(list_nodes, ['name', 'code'])

    def GenerateInit(self, prev_node, node, code_init):
        statement = '    self.' + node['name'] + ' = ' + node['code'] + '()'
        code_init.append(statement)

    def GenerateForward(self, prev_node, node, code_forward):
        return