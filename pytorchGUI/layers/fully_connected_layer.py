from validator_layer import RequiredNodeValidator

class FullyConnectedLayer(RequiredNodeValidator):
    def GetName(self):
        return "Fully Conn"

    def Validate(self, list_nodes):
        return RequiredNodeValidator.Validate(list_nodes, ['name', 'code', 'out_features'])

    def GenerateInit(self, prev_node, node, code_init):
        statement = '    self.' + node['name'] + ' = ' + node['code']
        statement += '('
        statement += 'in_features=' + str(prev_node['out_features'])
        statement += ', '
        statement += 'out_features=' + str(node['out_features'])
        statement += ')'
        code_init.append(statement)

    def GenerateForward(self, prev_node, node, code_forward):
        statement = '    x = self.'
        statement += node['name'] + '(x)'
        code_forward.append(statement)
