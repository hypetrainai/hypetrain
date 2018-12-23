from validator_layer import RequiredNodeValidator

class MaxPool2DLayer(RequiredNodeValidator):
    def GetName(self):
        return "Max Pool 2D"

    def Validate(self, list_nodes):
        return RequiredNodeValidator.Validate(list_nodes, ['kernel_size', 'name', 'code', 'stride'])

    def GenerateInit(self, prev_node, node, code_init):
        padding = int((node['kernel_size']-1)/2)
        statement = '    self.' + node['name'] + ' = ' + node['code']
        statement += '('
        statement += 'kernel_size=' + str(node['kernel_size'])
        statement += ', '
        statement += 'stride=' + str(node['stride'])
        statement += ', '
        statement += 'padding=' + str(padding)
        statement += ')'
        code_init.append(statement)

    def GenerateForward(self, prev_node, node, code_forward):
        statement = '    x = self.'
        statement += node['name'] + '(x)'
        code_forward.append(statement)
