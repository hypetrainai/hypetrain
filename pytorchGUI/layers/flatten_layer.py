from validator_layer import RequiredNodeValidator

class FlattenLayer(RequiredNodeValidator):
    def GetName(self):
        return "Flatten"

    def Validate(self, list_nodes):
        return RequiredNodeValidator.Validate(list_nodes, ['out_features'])

    def GenerateInit(self, prev_node, node, code_init):
        return

    def GenerateForward(self, prev_node, node, code_forward):
        statement = '    x = x.view(-1, ' + str(node['out_features']) + ')'
        code_forward.append(statement)
