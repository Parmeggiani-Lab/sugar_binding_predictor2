
path='../example/xylose/n_XYS_native/'



loc = path
exec(open('find_interacting_residue.py').read())
exec(open('keep_functional_group.py').read())
loc = path
exec(open('input_data_generate.py').read())


