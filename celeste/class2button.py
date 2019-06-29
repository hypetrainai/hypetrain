action_button_dict = {
    0: [''],
    1: ['a'],
    2: ['b'],
    3: ['rt'],
    4: ['a','b'],
    5: ['a','rt'],
    6: ['rt', 'b'],
    7: ['a','b','rt']
}

dpad_button_dict = {
    0: [''],
    1: ['r'],
    2: ['l'],
    3: ['u'],
    4: ['d'],
    5: ['r','u'],
    6: ['u','l'],
    7: ['l','d'],
    8: ['d','r']
}

class2button = {}

for action_key in action_button_dict:
    for dpad_key in dpad_button_dict:
        final_key = action_key*9+dpad_key
        if dpad_button_dict[dpad_key][0] == '':
            final_value = action_button_dict[action_key]
        elif action_button_dict[action_key][0] == '':
            final_value = dpad_button_dict[dpad_key]
        else:
            final_value = action_button_dict[action_key] + dpad_button_dict[dpad_key]
        class2button[final_key] = final_value