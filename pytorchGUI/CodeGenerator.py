

def GenerateCode(list_nodes):
    code_init = []
    code_forward = []
    code_init.append('class Test(nn.Module):')
    code_init.append('  def __init__(self):')
    code_init.append('    super(Test, self).__init__()')
    code_forward.append('  def forward(self,x):')
    for node in list_nodes:
        node.covered = False
    loss_node = FindLossNode(list_nodes)
    Generate_DFS(loss_node, code_init, code_forward)
    code_forward.append('    return x')

    init_lines = "\n".join(code_init)
    forward_lines = "\n".join(code_forward)
    all_lines = init_lines + "\n" + forward_lines
    return all_lines

def FindLossNode(list_nodes):
    for node in list_nodes:
        if node.type.name == "CE Loss":
            return node
    print("ERROR - Could not find CE Loss layer")
    return None

def Generate_DFS(node, code_init, code_forward):
    node.covered = True
    prev_node = None
    if len(node.prev) > 0:
        for prev in node.prev:
            if not prev.covered:
                Generate_DFS(prev, code_init, code_forward)
        prev_node = list(node.prev)[0]

    node.type.behavior.GenerateInit(prev_node, node, code_init)
    node.type.behavior.GenerateForward(prev_node, node, code_forward)
