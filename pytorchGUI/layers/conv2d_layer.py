from validator_layer import RequiredNodeValidator

class Conv2DLayer(RequiredNodeValidator):
    def GetName(self):
        return "Conv 2D"

    def Validate(self, list_nodes):
        return RequiredNodeValidator.Validate(list_nodes, ['kernel_size', 'name', 'code', 'out_channels', 'stride'])

    def GenerateInit(self, prev_node, node, code_init):
        padding = int((node['kernel_size']-1)/2)
        statement = '    self.' + node['name'] + ' = ' + node['code']
        statement += '('
        statement += 'in_channels=' + str(prev_node['out_channels'])
        statement += ', '
        statement += 'out_channels=' + str(node['out_channels'])
        statement += ', '
        statement += 'kernel_size=' + str(node['kernel_size'])
        statement += ', '
        statement += 'stride=' + str(node['stride'])
        statement += ', '
        statement += 'padding=' + str(padding)
        statement += ')'
        code_init.append(statement)

    def GenerateForward(self, prev_node, node, code_forward):
        statement = '    x = self.' + node['name'] + '(x)'
        code_forward.append(statement)