class RequiredNodeValidator():
    def Validate(list_nodes, required_nodes):
        requirements = {}
        for node in required_nodes:
            requirements[node] = False
        for node in list_nodes:
            if node.name in requirements:
                requirements[node.name] = True
        for name, exists in requirements.items():
            if not exists:
                return False, "Missing " + name
        
        return True, None