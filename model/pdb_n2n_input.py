import os,re,pymol,shutil,json,sys
from pymol import cmd,stored
import operator
import torch,math
import numpy as np
from measure_CH_pi import measure_CH_pi
from collections import Counter

from tqdm import tqdm

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))

def find_unique_loops(g_1):
    cycles = [path  for node in g_1 for path in dfs(g_1, node, node) if len(path)>2]
    [i.sort() for i in cycles]
    cycles=[list(x) for x in set(tuple(x) for x in cycles)]
    #print(cycles)
    return cycles

def piece_to_molecule(pieces_ori):
    print(pieces_ori)
    print(len(pieces_ori))
    pieces = [i for i in pieces_ori if i != 0]
    pieces = [i - 14 for i in pieces]
    #print(pieces)
    molecule_list = []
    molecule_idx = 0
    sugar_number=0
    for i in pieces:
        if i == 0:
            molecule_idx += 1
            sugar_number+=1
        elif i > 0:
            pass
        elif i < 0:
            molecule_idx += 1
        if molecule_idx>0:
            molecule_list.append(molecule_idx)
        else:
            molecule_list.append(1)

    return torch.LongTensor(molecule_list), sugar_number

def load_sugar_fragment_database():
    with open('sugar_fragment_type.txt', 'r') as fragment_type_file:
        fragment_type_line = fragment_type_file.readlines()
        fragment_type_file.close()

    sugar_atom2fragment = {}
    for i in fragment_type_line:
        try:
            sugar_atom2fragment[i.split('\t')[0].split()[0]] = [int(i.split('\t')[1].split()[0]), int(i.split('\t')[2].split()[0])]
        except IndexError:
            pass
    current_sugar_list=list(set([i.split('_')[0].split()[0] for i in sugar_atom2fragment]))
    return sugar_atom2fragment,current_sugar_list
def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    return edges


def detect_new_sugar(loc, pdb,sugar_atom2fragment,atom_type_dict,sugar_fragment_note_dict,save_exist_sugar=None,save_new_sugar_example=None,record_new_sugar=None,save_new_sugar_renaming=None,output_std_sugar_naming=None):

    cmd.reinitialize()
    cmd.load(loc+pdb,'pair_complex')
    with open(loc + pdb, 'r') as pdb_file:
        context_line = pdb_file.readlines()
        pdb_file.close()
    atom_line = [i for i in context_line if i[:4] == 'HETA' or i[:4] == 'ATOM']
    sugar_line = [i for i in context_line if i[:4] == 'HETA']
    # print([[i[31:38].split()[0],i[39:45].split()[0],i[47:56].split()[0]] for i in atom_line])
    atom_coord_list = np.array([[i[31:38].split()[0], i[39:45].split()[0], i[47:56].split()[0]] for i in
                                atom_line])  ##################################################
    atom_coord_list = atom_coord_list.astype(float)
    resi_atom_name = [i[17:21].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]
    #### check new sugar
    check_new_sugar = [
        [i[:4], i[7:11].split()[0], i[12:16].split()[0], i[17:21].split()[0], i[22:30].split()[0], i[77:78].split()[0]]
        for i in atom_line if i[17:21].split()[0] + '_' + i[12:16].split()[0] not in atom_type_dict]
    if save_exist_sugar != None:
        check_new_sugar = [
            [i[:4], i[7:11].split()[0], i[12:16].split()[0], i[17:21].split()[0], i[22:30].split()[0],
             i[77:78].split()[0]] for i in atom_line if
            i[:4] == 'HETA']
        new_sugar_dict={i[7:11].split()[0]:[i[:4], i[7:11].split()[0], i[12:16].split()[0], i[17:21].split()[0], i[22:30].split()[0],
             i[77:78].split()[0]] for i in atom_line if
            i[:4] == 'HETA'}
    if check_new_sugar != []:
        enablePrint()
        print('new sugar type found:', list(set([i[0] + '_' + i[3] for i in check_new_sugar])), 'in ', pdb)
        link_line = [i for i in context_line if i[:4] == 'CONE']
        new_sugar_residx_atomidx = {}  ### new sugar id - atom id
        for i in check_new_sugar:
            new_sugar_residx_atomidx[i[1]] = i[4]
        print(new_sugar_residx_atomidx)
        link_record_dict = {}
        for i in link_line:
            atom1 = i.split()[1]
            atom_tar = [atom for atom in i.split()[2:] if atom in new_sugar_residx_atomidx]
            if atom_tar != [] and atom1 in new_sugar_residx_atomidx:
                link_record_dict[atom1] = atom_tar

        for node in link_record_dict:
            for neigbhour in link_record_dict[node]:
                if (link_record_dict[node][-1]==link_record_dict[neigbhour][-1]=='O'):
                    link_record_dict[node] = [x for x in link_record_dict[node] if x != neigbhour]
        loops_in_graph = find_unique_loops(link_record_dict)  ### find loop in linking record
        # print(loops_in_graph)
        new_element_type_atomidx = {}  ###  element- atom id
        for i in check_new_sugar:
            new_element_type_atomidx[i[1]] = i[5]
        new_sugar_atom_name_atomidx = {}  ### new sugar name-atom name - atom id
        for i in check_new_sugar:
            new_sugar_atom_name_atomidx[i[1]] = [i[3], i[2]]
        loops_atom_in_sugar = []  ### check loops are in one residue
        loops_in_sugar = []
        # print(check_new_sugar[0])
        print('found loops: ')
        for loop in loops_in_graph:
            print(loop)
            check_loop_in_molecule = [new_sugar_residx_atomidx[i] for i in loop]
            if len(list(set(check_loop_in_molecule))) == 1:
                loops_atom_in_sugar.extend(loop)
                loops_in_sugar.append(loop)
        # print(loops_atom_in_sugar)
        new_sugar_fragment_type_info = {}
        new_sugar_atom_type_info = {}

        new_sugar_node_name_dict = {}
        new_sugar_fragment_type_naming_dict = {}
        new_sugar_atom_type_naming_dict = {}
        ### for each atom in new sugar , find proper atom type and fragment
        for i in check_new_sugar:

            defined_atom_type = ''
            defined_fragment_type = ''
            i_residue_atomname = [i[3], i[2]]
            i_element = i[5]
            i_atomidx = i[1]
            neigbor_idx_list = link_record_dict[i_atomidx]
            loop_neigbor_idx = [atom for atom in link_record_dict[i_atomidx] if atom in loops_atom_in_sugar]
            # not_loop_neigbor_resi_atom_name_list = [new_sugar_atom_name_atomidx[atom] for atom in link_record_dict[i_atomidx] if atom not in loops_atom_in_sugar]
            not_loop_neigbor_element_list = [new_element_type_atomidx[atom] for atom in link_record_dict[i_atomidx] if
                                             atom not in loops_atom_in_sugar]
            not_loop_neigbor_in_loop_list = [atom in loops_atom_in_sugar for atom in link_record_dict[i_atomidx] if
                                             atom not in loops_atom_in_sugar]
            not_loop_neigbor_neigbor_idx_list = [link_record_dict[atom] for atom in link_record_dict[i_atomidx] if
                                                 atom not in loops_atom_in_sugar]
            not_loop_neigbor_neigbor_element_list = []
            for n1 in not_loop_neigbor_neigbor_idx_list:
                not_loop_neigbor_neigbor_element_list.append([new_element_type_atomidx[n2] for n2 in n1])
            if not_loop_neigbor_neigbor_element_list == []:
                not_loop_neigbor_neigbor_element_list = [[]]
            neigbor_number = len(neigbor_idx_list)
            #### element type
            if i_element == 'C':
                if i_atomidx in loops_atom_in_sugar:  #### if C in loop
                    loop_len = [len(loop) for loop in loops_in_sugar if i_atomidx in loop][0]
                    defined_atom_type = 'CH' + str(4 - neigbor_number)
                    defined_fragment_type = [14, loop_len]
                if Counter(not_loop_neigbor_element_list)['O'] == 2:  #### if C not in loop and connect to 2 O, then COO
                    defined_atom_type = 'COO'
                    defined_fragment_type = [18, 3]
                if Counter(not_loop_neigbor_element_list)['O'] == 1 and len(
                        loop_neigbor_idx) <= 1:  #### if C connect to 1 O and loop or not, then C-O
                    defined_atom_type = 'CH' + str(4 - neigbor_number)
                    defined_fragment_type = [16, 2]
                if Counter(not_loop_neigbor_element_list)['O'] == 1 and Counter(not_loop_neigbor_element_list)[
                    'N'] == 1 and Counter(not_loop_neigbor_element_list)['C'] == 1:  ## amide sp2 C
                    defined_atom_type = 'CNH2'
                    defined_fragment_type = [17, 4]
                if defined_atom_type == '':
                    defined_atom_type = 'CH' + str(4 - neigbor_number)
                    if Counter(not_loop_neigbor_neigbor_element_list[0])['O'] == 1 and \
                            Counter(not_loop_neigbor_neigbor_element_list[0])['N'] == 1 and \
                            Counter(not_loop_neigbor_neigbor_element_list[0])['C'] == 1:
                        defined_fragment_type = [17, 4]  ## amide sp3 C
                    else:
                        defined_fragment_type = [19, 1]  ## singe C
            elif i_element == 'N':
                if Counter(not_loop_neigbor_neigbor_element_list[0])['O'] == 1 and \
                        Counter(not_loop_neigbor_neigbor_element_list[0])['N'] == 1 and \
                        Counter(not_loop_neigbor_neigbor_element_list[0])['C'] == 1:
                    defined_atom_type = 'NH2O'
                    defined_fragment_type = [17, 4]  ## amide sp2 N
                if not_loop_neigbor_element_list == ['S']:  ## N-SO4
                    defined_atom_type = 'NtrR'
                    defined_fragment_type = [21, 1]
            elif i_element == 'S':
                if Counter(not_loop_neigbor_element_list)['O'] >= 3:  #### -SO3
                    defined_atom_type = 'SO4'
                    defined_fragment_type = [22, 4]
            elif i_element == 'P':
                if Counter(not_loop_neigbor_element_list)['O'] >= 3:  #### -PO3
                    defined_atom_type = 'Phos'
                    defined_fragment_type = [20, 4]
            elif i_element == 'O':
                if i_atomidx in loops_atom_in_sugar:  #### O in loop
                    loop_len = [len(loop) for loop in loops_in_sugar if i_atomidx in loop][0]
                    defined_atom_type = 'OS'
                    defined_fragment_type = [14, loop_len]

                ## O neighbour has another N nad C
                if Counter(not_loop_neigbor_neigbor_element_list[0])['O'] == 1 and \
                        Counter(not_loop_neigbor_neigbor_element_list[0])['N'] == 1 and \
                        Counter(not_loop_neigbor_neigbor_element_list[0])['C'] == 1:
                    defined_atom_type = 'ONH2'
                    defined_fragment_type = [17, 4]  ## amide sp2 O
                if Counter(not_loop_neigbor_neigbor_element_list[0])['O'] >= 2:  ## neigbour connect to 2 O
                    defined_atom_type = 'OOC'
                    if not_loop_neigbor_element_list == ['S']:
                        defined_fragment_type = [22, 4]  ## SO4
                    if not_loop_neigbor_element_list == ['C']:
                        defined_fragment_type = [18, 3]  ## COO
                    if not_loop_neigbor_element_list == ['P']:
                        defined_fragment_type = [20, 4]  ## PO4

                if defined_atom_type == '' or defined_fragment_type=='':
                    defined_atom_type = 'OH'
                    if not_loop_neigbor_element_list == ['C']:
                        defined_fragment_type = [16, 2]  ## C-O
                    else:
                        defined_fragment_type = [15, 1]  # -OH

            if defined_atom_type == '' or defined_fragment_type == '':
                print('undefined atom :', i_atomidx, i_element, i_residue_atomname, i_atomidx in loops_atom_in_sugar,
                      len(loop_neigbor_idx), not_loop_neigbor_element_list, not_loop_neigbor_in_loop_list,
                      Counter(not_loop_neigbor_element_list)['O'],
                      not_loop_neigbor_neigbor_element_list)
                print('\t', defined_atom_type, defined_fragment_type)  # ,

            new_sugar_atom_type_info['_'.join(i_residue_atomname)] = defined_atom_type
            new_sugar_fragment_type_info['_'.join(i_residue_atomname)] = defined_fragment_type

            new_sugar_atom_type_naming_dict[i_atomidx] = defined_atom_type
            new_sugar_fragment_type_naming_dict[i_atomidx] = defined_fragment_type
            new_sugar_node_name_dict[i_atomidx] = i_residue_atomname[1]
            #
            #

        ### add new atom in to fragment/atom type dict
        print('new defined atom added: ', len(new_sugar_fragment_type_info), len(new_sugar_atom_type_info))
        for i in new_sugar_fragment_type_info:
            sugar_atom2fragment[i] = new_sugar_fragment_type_info[i]
        for i in new_sugar_atom_type_info:
            atom_type_dict[i] = new_sugar_atom_type_info[i]

        if save_new_sugar_example != None:
            cmd.remove('polymer')
            cmd.zoom('not polymer')
            cmd.scene('no_label', 'store', )

            cmd.label('not polymer', 'name')
            cmd.scene('atom_name', 'store', )
            cmd.scene('no_label', )

            cmd.split_states('not polymer and pair_complex', prefix='fragment_')
            # new_sugar_resi_list=list(set([i.split('_')[0] for i in new_sugar_atom_type_info.keys()]))
            # print('new_sugar_resi_list:',new_sugar_resi_list)
            # for new_sugar_resi in new_sugar_resi_list:
            # print(new_sugar_resi)

            g_info=([link_record_dict, new_sugar_atom_type_naming_dict, new_sugar_fragment_type_naming_dict,
                   new_sugar_node_name_dict])

            ### output new sugar type naming info
            if output_std_sugar_naming != None:
                with open(loc + pdb[:-4] + '_new_sugar_naming.txt', 'w') as f:
                    f.write(pdb[:-4] + '\t' + str([(link_record_dict),
                                                   (new_sugar_atom_type_naming_dict),
                                                   (new_sugar_fragment_type_naming_dict),
                                                   (new_sugar_node_name_dict), ]) + '\n')
                    f.close()

            print('new_sugar_fragment_type_info', )
            for i in new_sugar_fragment_type_info:
                # if i.split('_')[0] == new_sugar_resi:
                print(i, new_sugar_fragment_type_info[i])
                cmd.label('fragment_0001 and /////' + i.split('_')[1],
                          '"' + sugar_fragment_note_dict[new_sugar_fragment_type_info[i][0]] + '"')
            cmd.hide('everything', 'not fragment_0001')
            cmd.scene('fragment_label', 'store', )
            cmd.scene('no_label')

            cmd.split_states('not polymer and pair_complex', prefix='atom_')
            print('new_sugar_atom_type_info', )
            for i in new_sugar_atom_type_info:
                # if i.split('_')[0] == new_sugar_resi:
                #print(i, new_sugar_atom_type_info[i])
                cmd.label('atom_0001 and /////' + i.split('_')[1], '"' + new_sugar_atom_type_info[i] + '"')
            cmd.hide('everything', 'not atom_0001')
            cmd.scene('atom_label', 'store', )
            cmd.save(loc + pdb[:-4] + '_new_sugar.pse')





        if record_new_sugar == None:
            print('Skip new sugar adding to database..')
        else:
            print('Appending new sugar to database..')
            print('new_sugar_fragment_type_info', new_sugar_fragment_type_info)
            with open('sugar_fragment_type.txt', 'a') as append_new_define_fragement:
                append_new_define_fragement.write('\n')
                for i in new_sugar_fragment_type_info:
                    append_new_define_fragement.write('{}\t{}\t{}   \n'.format(i, new_sugar_fragment_type_info[i][0],
                                                                               new_sugar_fragment_type_info[i][1]))
                append_new_define_fragement.close()
            print('new_sugar_atom_type_info', new_sugar_atom_type_info)
            with open('atom_type_old.txt', 'a') as append_new_define_atom:
                append_new_define_atom.write('\n')
                for i in new_sugar_atom_type_info:
                    append_new_define_atom.write(
                        '{} {} {}   \n'.format(i.split('_')[0], i.split('_')[1], new_sugar_atom_type_info[i]))
                append_new_define_atom.close()

        if save_new_sugar_renaming!=None:
            return g_info



def data_preprocessing(loc,pdb,output_txt=1,output_CH_pi_record=1,max_len_seq=0,batch_size=1,ignore_CH_pi=0,
                       generate_file_only=0,debug_mode=0,out_path=None,print_info=None,record_new_sugar=None,save_new_sugar_example=None,save_exist_sugar=None,save_new_sugar_renaming=None,output_std_sugar_naming=None):
   if 'env_prepare'!=1:
       ### frament atom of residue
       """
       atom to keep:
           1. sugar main ring + rest of sugar
           2. Chi-1 functional group of side-chain of N/O/S interacting amino acid (at least two heavy atom)
           3. N/O interacting main chain (alter residue name to GLY)
           4. aromatic sidechain with CH-pi score record

       info to keep:
           1. atom coordinates
           2. atom type
           3. fragment type
           4. adjacent matrix
           5. bond type

       """
       aa_keep_atom_dict = {
           'SER': 'OG+CB',
           'THR': 'OG1+CB',
           'ASN': 'OD1+ND2+CG',
           'GLN': 'OE1+NE2+CD',
           'CYS': 'CB+SG',
           'PRO': 'CA+N+CD+CG+CB',
           'MET': 'CG+SD',
           'PHE': 'CG+CD1+CD2+CE1+CE2+CZ',
           'TYR': 'CG+CD1+CD2+CE1+CE2+CZ+OH',
           'TRP': 'CG+CD1+NE1+CE2+CD2+CE3+CZ3+CH2+CZ2',
           'ARG': 'NH2+NH1+CZ',
           'HIS': 'CG+ND1+CE1+NE2+CD2',
           'LYS': 'NZ+CE',
           'ASP': 'CG+OD1+OD2',
           'GLU': 'CD+OE1+OE2',

       }
       ############## process atom type
       with open('atom_type_old.txt', 'r') as atom_type_file:
           atom_type_line = atom_type_file.readlines()
           atom_type_file.close()
       atom_type_dict = {}
       for i in atom_type_line:
           try:
               atom_type_dict[i.split()[0] + '_' + i.split()[1]] = i.split()[2]
           except IndexError:
               pass
       current_resi_atom_name_list = []
       for i in atom_type_dict:
           current_resi_atom_name_list.append(atom_type_dict[i])
       current_atomtype_list = list(set(current_resi_atom_name_list))
       # print('There are ',len(current_atomtype_list),' residue-atom names in  dict,')
       # print('with ',len(list(set(current_atomtype_list))),' unique atomtypes in dict.\n')
       unq_atomtype_list = (list(set(current_atomtype_list)))

       atomtype2id = {
           'CAbb': 1,  ## Ca sp3 for backbone
           'CH0': 2,  ## C sp3 0-H
           'CH1': 3,  ## C sp3 1-H
           'CH2': 4,  ## C sp3 2-H
           'CH3': 5,  ## C sp3 3-H
           'CNH2': 6,  ## C sp2 -N-C(=O)-R Amide C
           'COO': 7,  ## C sp2 -COO carboxyl C
           'CObb': 8,  ## C sp3 for backbone
           'aroC': 25,  ### C sp3 aromatic C

           'NH2O': 9,  ## N sp2 -N-C(=O)-R Amide N
           'Narg': 10,  ## N sp3 Arg terminal N only currently
           'Nbb': 11,  ## N sp2 for backbone
           'Nhis': 12,  ## N sp2 His N with positive changed only currently
           'Nlys': 13,  ## N sp3 Lys terminal N with positive changedonly currently
           'Npro': 14,  ## N sp3? Pro N only currently
           'NtrR': 15,  ## N sp3 Arg connecting N  (C-N-C) or (C-N-SO3)
           'Ntrp': 16,  ## N sp2 Trp/His-NE2 N only currently

           'OCbb': 17,  ## O for backbone
           'OH': 18,  ## O sp3 for hydroxyl, acid
           'ONH2': 19,  ## O sp2 -N-C(=O)-R Amide O
           'OOC': 20,  ## O sp2 -COO carboxyl O
           'OS': 21,  ## O sp3 for C-O-C, and sugar ring O

           'S': 22,  ## S sp3  C-S-C  for MET only
           'SH1': 23,  ## S sp3 terminal S for CYS/CYZ
           'SO4': 27,  ## S sp3  S04

           'VIRT': 24,
           'Phos': 26,  ## P sp3  P04

       }

       aa2fid = {
           'SER': 1,  # Ser_Thr
           'THR': 1,  # Ser_Thr
           'ASN': 2,  # Asn_Gln
           'GLN': 2,  # Asn_Gln
           'CYS': 3,  # Cys_Met
           'PRO': 4,
           'MET': 3,  # Cys_Met
           'PHE': 5,
           'TYR': 6,
           'TRP': 7,
           'ARG': 8,
           'HIS': 9,
           'LYS': 10,
           'ASP': 11,  # Asp_Glu
           'GLU': 11,  # Asp_Glu
           'GLY': 12,  # aa backbone
       }
       '''
       <s>=0,<pad>=13
       14	 6/5	C5O1/C4O1  # 6/5-atom sugar ring 
       15	 1	    O1 # Oxygen
       16	 2	    C1O1 # C-O
       17	 4	    C2O1N1 # amide
       18	 3	    CO2 #COO
       19	 1	    C1 # Carbon
       20	 4	    O3P # PO3 of -O-PO3
       21	 1	    N1 # Ntrp sp2 (improperly use, mix with sp3+sp2)
       22	 4	    O3S # SO3 of -O-SO3

       '''
       sugar_fragment_note_dict={
       14:	 'sugar_ring',
       15:   'Oxygen',
       16:   'C-O',
       17:   'amide',
       18:   'COO',
       19:   'Carbon',
       20:   'PO3',
       21:   'Ntrp',
       22:   'SO3',
       }


       ############## process sugar fragment type
       sugar_atom2fragment, current_sugar_list = load_sugar_fragment_database()
       aromatic_ring_atom_dict = {
           'PHE': ['CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
           'TYR': ['CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
           'TRP': ['NE1', 'CD1', 'CG', 'CD2', 'CE2'],
           'TRB': ['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'],
           'HIS': ['CD2', 'CG', 'CE1', 'ND1', 'NE2'],
       }
   try:
        if 'pdb to elicit pdb'!=1:
            if print_info == None:
                blockPrint()
            if out_path==None:
                out_path=loc

            if ignore_CH_pi != 1:
                out_file_name = pdb[:-4] + '_elicit_info.txt'
            else:
                out_file_name = pdb[:-4] + '_elicit_no_CH_pi_info.txt'

            if os.path.exists(out_path+out_file_name)==1 and generate_file_only==1:
                return 'file already exist'

            ##### replace any _ into x in .pdb file

            with open(loc+pdb,'r') as f:
                ori_file_context=f.read()
                to_replace = re.findall('_', ori_file_context)
                f.close()
            if len(to_replace)>0:
                ori_file_context=ori_file_context.replace('_','x')
                with open(loc+pdb,'w') as f:
                    f.write(ori_file_context)
                    f.close()

            CH_pi_info_dict = {}
            if ignore_CH_pi!=1:
                CH_pi_record=measure_CH_pi(loc, pdb,out_record=output_CH_pi_record,out_path=out_path,print_info=print_info)
                if CH_pi_record=='cannot_load_file':
                    enablePrint()
                    print('Cannot load '+loc+pdb)
                    if print_info == None:
                        blockPrint()
                    return 'Cannot load '+loc+pdb
                CH_pi_info_dict[pdb[:-4]]=CH_pi_record
            else:
                CH_pi_info_dict[pdb[:-4]]=''



            cmd.reinitialize()
            cmd.load(loc + pdb,'pair_complex')
            cmd.remove('hydrogen')
            #cmd.remove("(not polymer)")
            cmd.remove('/////OXT')
            cmd.remove('////F7W')
            cmd.alter('all',"alt=''")

            #### find interacting residue
            try:
                CH_pi_info=CH_pi_info_dict[pdb[:-4]]
                list_of_ch_pi=[]
                for i in CH_pi_info:
                    if i.split('_')[1]=='TRB':
                        list_of_ch_pi.append('////' + 'TRP' + '`' + str(int(i.split('_')[0])-0))
                    else:
                        list_of_ch_pi.append('////' + i.split('_')[1] + '`' + str(int(i.split('_')[0])-0))


                CH_pi_residue=' or '.join(list_of_ch_pi)
                CH_pi_residue=CH_pi_residue.replace('-','\-')
            except KeyError:
                CH_pi_residue=''

            #blockPrint()
            print('\tCH-pi residue',CH_pi_residue)


            cmd.select('atom_NOS','symbol O+N+S')
            #### 2024.12.10 add TRP in porlar interaction
            cmd.select('polar_charged','(byres( ((not polymer) and pair_complex) around 3.5) and atom_NO) and (////ARG or ////HIS or ////LYS or ////ASP or ////GLU or ////Ser or ////THR or ////ASN or ////GLN or ////TYR or ////TRP)')
            #cmd.save(loc+'test.pse')
            cmd.remove('metal') ### ignore ion for design using


            if CH_pi_residue!='':
                cmd.select('aromatic',CH_pi_residue+' and ((not polymer) around 10)')
                take_chi_pi_record = '1'
            else:
                cmd.select('aromatic', 'None')
                take_chi_pi_record = '0'

            cmd.remove('not (not polymer or polar_charged or aromatic)')
            resi_num = cmd.select('ca', '/////CA')



            cmd.save(loc + pdb[:-4] + '_resi.pdb')
                #cmd.save(loc + pdb[:-4] + '_resi.pse')
            if resi_num == 0 and save_new_sugar_example==None:
                print('No interacting residue')
                return 'No interacting residue'

            ### keep functional group

            cmd.select('CH_pi_side', 'None')
            cmd.select('CH_pi_sugar', 'None')
            try:
                CH_pi_info = CH_pi_info_dict[pdb[:-4]]
                CH_pi_residue = '+'.join([i.split('_')[0] for i in CH_pi_info])
                CH_pi_sugar = '+'.join([i.split('_')[2] for i in CH_pi_info])
                # print(CH_pi_info)
                # print(CH_pi_residue,CH_pi_sugar)
                if CH_pi_residue != '':
                    cmd.select('CH_pi_side', 'sidechain and i. ' + CH_pi_residue + '')
                    cmd.select('CH_pi_sugar', 'i. ' + CH_pi_sugar + '')
            except KeyError:
                pass
            # if print(cmd.select('////PRO')!=0):
            #    print(pdb)


            cmd.select('sc_1', 'byres((not polymer around 3.5) and sidechain) ')
            cmd.select('sc_interacting', 'sc_1 and sidechain')
            cmd.select('bb_1', 'byres((not polymer around 3.5) and backbone) ')
            cmd.select('bb_interacting', 'bb_1 and backbone')


            cmd.remove('not ( (not polymer) or sc_interacting or bb_interacting or CH_pi_side) or hydrogen')
            # cmd.save(loc_out+pdb[:-8]+'elicit.pse')
            for aa in aa_keep_atom_dict:
                # print(aa,aa_keep_atom_dict[aa])
                cmd.remove('(sc_interacting or CH_pi_side) and ////' + aa + ' and (not n. ' + aa_keep_atom_dict[aa] + ')')
            cmd.alter('bb_interacting', 'resn="GLY"')

            ##### keep only interacting sugar

            cmd.select('inter_sugar', 'byres(polymer around 3.5)')
            cmd.remove('(not polymer) and (not inter_sugar) and (not CH_pi_sugar)')

            cmd.set('pdb_conect_all', 'on')
            if (cmd.select('not polymer') == 0 or cmd.select('polymer') == 0) and save_new_sugar_example==None:
                print('No interacting fragment')
                return 'No interacting fragment'



            if ignore_CH_pi!=1:
                elict_file_name= pdb[:-4] + '_elicit.pdb'
            else:
                elict_file_name = pdb[:-4] + '_elicit_no_CH_pi.pdb'

            cmd.alter('all','segi=""')
            ### sort sugar before protein
            cmd.alter('polymer', 'segi="B"')
            cmd.alter('not polymer', 'segi="A"')
            cmd.sort()
            cmd.save(loc+elict_file_name)
            if save_new_sugar_example != None:
                cmd.reinitialize()
                cmd.load(loc + pdb[:-4] + '_resi.pdb','pair_complex')
                cmd.remove('polymer')
                cmd.remove('ion')
                cmd.save(loc+elict_file_name)
        #### generate input
        with open(loc + elict_file_name, 'r') as pdb_file:
            context_line = pdb_file.readlines()
            pdb_file.close()
        atom_line = [i for i in context_line if i[:4] == 'HETA' or i[:4] == 'ATOM']
        sugar_line = [i for i in context_line if i[:4] == 'HETA']
        if sugar_line == [] :
            print( 'No recognized sugar')
            return 'No recognized sugar'
        print('atom number: ', len(atom_line))
        # print([[i[31:38].split()[0],i[39:45].split()[0],i[47:56].split()[0]] for i in atom_line])
        atom_coord_list = np.array([[i[31:38].split()[0], i[39:45].split()[0], i[47:56].split()[0]] for i in
                                    atom_line])  ##################################################
        atom_coord_list = atom_coord_list.astype(float)
        resi_atom_name = [i[17:21].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]

        if 'detect_sugar'!=1:
            check_new_sugar=[[i[:4],i[7:11].split()[0],i[12:16].split()[0],i[17:21].split()[0],i[22:30].split()[0],i[77:78].split()[0]] for i in atom_line if i[17:21].split()[0] + '_' + i[12:16].split()[0] not in atom_type_dict]
            if check_new_sugar != [] or save_exist_sugar!=None:
                detect_new_sugar(loc, elict_file_name,sugar_atom2fragment,atom_type_dict,sugar_fragment_note_dict,save_exist_sugar=save_exist_sugar,save_new_sugar_example=save_new_sugar_example,record_new_sugar=record_new_sugar,save_new_sugar_renaming=save_new_sugar_renaming,output_std_sugar_naming=output_std_sugar_naming)
            if print_info == None:
                blockPrint()
            if debug_mode == 0:
                os.remove(loc + elict_file_name)
                os.remove(loc + pdb[:-4] + '_resi.pdb')

        if 'generate node info' != 1:
            print('\ngenerate node info...\n')
            atomtype_list = [atom_type_dict[i] for i in resi_atom_name]  #########################################
            print('resi_atom_name',resi_atom_name)
            atomtype_id_list = [atomtype2id[atom_type_dict[i]] for i in resi_atom_name]  ##################
            print('atomtype involved: ', len(list(set(atomtype_list))))
            polartype_list = []  #############################
            for i in atomtype_list:
                if i[0] == 'N' or i[0] == 'O' or i[0] == 'S':
                    polartype_list.append('polar')
                else:
                    polartype_list.append('in-polar')
            residue_id_list = [i[17:21].split()[0] + '_' + i[22:30].split()[0] for i in atom_line]  ###########################
            print('residue_id_list',residue_id_list)

            is_sugar_list = [1 if i[:4]=='HETA' else 0 for i  in atom_line ]

            atom_to_mole_id_list = [] ################################
            detected_list = []
            mole_index = 0
            for i in residue_id_list:
                if i not in detected_list:
                    detected_list.append(i)
                    mole_index += 1
                atom_to_mole_id_list.append(mole_index)


            print('residues involved: ', len(list(set(residue_id_list))))

            atom_resi_id_list = [i[17:21].split()[0] + '_' + i[22:30].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]
            print('atom_resi_id_list',atom_resi_id_list)
            #print((set(atom_resi_id_list)))
            if len(atom_resi_id_list) != len(set(atom_resi_id_list)):
                return 'duplicate atom in data-preprocessing, skip'

            atom_resi_id_list_for_sugar = [i[22:30].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]
            #print(atom_resi_id_list_for_sugar)

            atompiece_list = []  ###### what type of fragment atom belong to
            for i in resi_atom_name:
                if i[:3] not in aa2fid:
                    #print(i,sugar_atom2fragment[i])
                    atompiece_list.append(sugar_atom2fragment[i][0]) ### fragment type not defined
                else:
                    atompiece_list.append(aa2fid[i[:3]])
            # print(atompiece_list)
            # print(atomtype_list)

            atompos_list = []  ###### which fragment in molecule atom belong to, generate piece order, need sugar index first then residue index
            piece_list = []
            piece_id = 0
            piece_type_b4 = ''
            fragment_store = {}
            for i, resi_id in enumerate(residue_id_list):

                piece_id_b4 = piece_id

                if residue_id_list[i] != residue_id_list[i - 1]:
                    # print(residue_id_list[i])
                    if i > 0:
                        if resi_id[:3] not in aa2fid:
                            pass
                        else:
                            # atompos_list.append(piece_id)

                            piece_list.append(aa2fid[residue_id_list[i][:3]])
                            if piece_id < piece_id_b4 + 1:
                                piece_id += 1

                if resi_id[:3] not in aa2fid:
                    piece_type_id = sugar_atom2fragment[resi_atom_name[i]]
                    if (piece_type_id[0] not in fragment_store) or (fragment_store[piece_type_id[0]][0] == 0):
                        fragment_remain = piece_type_id[1]
                        if i == 0 or piece_type_b4 == 'resi':
                            piece_id += 1
                            atompos_list.append(piece_id)
                            fragment_store[piece_type_id[0]] = [fragment_remain - 1, 1]

                        else:
                            atompos_list.append(piece_id)
                            fragment_store[piece_type_id[0]] = [fragment_remain - 1, piece_id]

                        piece_id += 1
                        piece_list.append(piece_type_id[0])
                    else:
                        fragment_store[piece_type_id[0]][0] -= 1
                        atompos_list.append(fragment_store[piece_type_id[0]][1])
                    piece_type_b4 = 'sugar'
                    # print(fragment_store,piece_id)
                else:
                    piece_type_b4 = 'resi'
                    try:
                        if piece_id > atompos_list[-1] + 1:
                            piece_id -= 1
                        atompos_list.append(piece_id)
                    except IndexError:
                        pass
                # print(piece_id,piece_type_b4)
            piece_list.insert(0, 0)
            piece_list.append(0)
            # print(piece_list)
            if len(piece_list) > max_len_seq:
                max_len_seq = len(piece_list)
            #len_seq_record.append(len(piece_list))

        if 'generate edge info' != 1:
            print('\ngenerate edge info...\n')
            ############# edge info
            try:
                CH_pi_info = CH_pi_info_dict[pdb[:-4]]
            except KeyError:
                CH_pi_info = []
            print(CH_pi_info)

            print('CH-pi interaction number: ', len(CH_pi_info))
            CH_pi_interaction_info = []
            for i in CH_pi_info:
                ring_id = i.split('_')[1]
                if i.split('_')[1] == 'TRB':
                    ring_atom_id_list = [atom_resi_id_list.index('TRP' + '_' + str(int(i.split('_')[0]) - 0) + '_' + j) + 1 for
                                         j in
                                         aromatic_ring_atom_dict[ring_id]]
                else:
                    ring_atom_id_list = [
                        atom_resi_id_list.index(i.split('_')[1] + '_' + str(int(i.split('_')[0]) - 0) + '_' + j) + 1 for j in
                        aromatic_ring_atom_dict[ring_id]]
                print('ring_atom_id_list',ring_atom_id_list)
                if i.split('_')[2] == 'B':
                    sugar_C_id = atom_resi_id_list_for_sugar.index(i.split('_')[3] + '_' + i.split('_')[4]) + 1
                else:
                    sugar_C_id = atom_resi_id_list_for_sugar.index(i.split('_')[2] + '_' + i.split('_')[3]) + 1
                # print(sugar_C_id)
                for j in ring_atom_id_list:
                    CH_pi_interaction_info.append([sugar_C_id, j])
                    CH_pi_interaction_info.append([j, sugar_C_id])
            print(CH_pi_interaction_info)

            bond_line = [i for i in context_line if i[:4] == 'CONE']
            bond_pair_info = []
            for i in bond_line:
                bond_line_info = i.split()
                for j in range(len(bond_line_info) - 2):
                    bond_pair_info.append([int(bond_line_info[1]), int(bond_line_info[j + 2])])
            print('covalent bonds number: ', len(bond_pair_info) / 2)

            pairwise_matrix = get_edges(len(atom_line))  # length = #atom^2-#atom #######################################

            atom_resi_id_polar_interaction_count=[0 for i in atom_resi_id_list]
            atom_resi_id_inpolar_interaction_count=[0 for i in atom_resi_id_list]
            bond_list = []  #######################
            contact_list = []  ##################
            polar_contact_list=[]
            inpolar_contact_list=[]
            CH_pi_list = []
            aa_contact_list = []  ###################
            polar_aa_contact_list = []
            inpolar_aa_contact_list = []
            edge_select_list = []
            edge_select = torch.zeros(batch_size, len(atomtype_id_list), len(atomtype_id_list)).to(
                torch.int32)  ##########################
            print(edge_select.size())
            print('atompos_list', len(atompos_list))
            print('atom_line', len(atom_line))

            for i in range(len(pairwise_matrix[0])):
                if [int(pairwise_matrix[0][i]) + 1, int(pairwise_matrix[1][i]) + 1] in bond_pair_info:
                    bond_list.append(1)
                else:
                    bond_list.append(0)

                if [int(pairwise_matrix[0][i]) + 1, int(pairwise_matrix[1][i]) + 1] in CH_pi_interaction_info:
                    CH_pi_list.append(1)
                else:
                    CH_pi_list.append(0)

                if math.dist(atom_coord_list[pairwise_matrix[0][i]], atom_coord_list[pairwise_matrix[1][i]]) < 3.5 and \
                        bond_list[i] == 0 and residue_id_list[pairwise_matrix[0][i]] != residue_id_list[
                    pairwise_matrix[1][i]] and polartype_list[pairwise_matrix[0][i]] == polartype_list[pairwise_matrix[1][i]]:
                    contact_list.append(1)
                    if polartype_list[pairwise_matrix[0][i]] =='polar':
                        polar_contact_list.append(1)
                        inpolar_contact_list.append(0)
                        atom_resi_id_polar_interaction_count[pairwise_matrix[0][i]]+=1
                        atom_resi_id_polar_interaction_count[pairwise_matrix[1][i]] += 1
                    elif polartype_list[pairwise_matrix[0][i]] != 'polar':
                        inpolar_contact_list.append(1)
                        polar_contact_list.append(0)
                        atom_resi_id_inpolar_interaction_count[pairwise_matrix[0][i]] += 1
                        atom_resi_id_inpolar_interaction_count[pairwise_matrix[1][i]] += 1
                    print(polartype_list[pairwise_matrix[0][i]],atom_resi_id_list[pairwise_matrix[0][i]],atom_resi_id_list[pairwise_matrix[1][i]])
                    # print(int(pairwise_matrix[0][i])+1,int(pairwise_matrix[1][i])+1)
                    if (residue_id_list[pairwise_matrix[0][i]].split('_')[0] not in current_sugar_list) and (
                            residue_id_list[pairwise_matrix[1][i]].split('_')[0] not in current_sugar_list):
                        aa_contact_list.append(1)
                        if polartype_list[pairwise_matrix[0][i]] == 'polar':
                            polar_aa_contact_list.append(1)
                            inpolar_aa_contact_list.append(0)
                        elif polartype_list[pairwise_matrix[0][i]] != 'polar':
                            inpolar_aa_contact_list.append(1)
                            polar_aa_contact_list.append(0)
                        # print(int(pairwise_matrix[0][i]) + 1, int(pairwise_matrix[1][i]) + 1)
                    else:
                        aa_contact_list.append(0)
                        inpolar_aa_contact_list.append(0)
                        polar_aa_contact_list.append(0)
                else:
                    contact_list.append(0)
                    aa_contact_list.append(0)
                    inpolar_aa_contact_list.append(0)
                    polar_aa_contact_list.append(0)
                    polar_contact_list.append(0)
                    inpolar_contact_list.append(0)

                #  if error here, sort sugar index before residue index
                # print(atompos_list[pairwise_matrix[1][i]],atompos_list[pairwise_matrix[0][i]])
                # print(atom_coord_list[pairwise_matrix[0][i]],atom_coord_list[pairwise_matrix[1][i]])
                try:
                    if atompos_list[pairwise_matrix[0][i]] != atompos_list[pairwise_matrix[1][i]] and math.dist(
                            atom_coord_list[pairwise_matrix[0][i]], atom_coord_list[pairwise_matrix[1][i]]) < 3.5:
                        edge_select_list.append(1)
                        edge_select[batch_size - 1][pairwise_matrix[0][i]][pairwise_matrix[1][i]] = 1
                    else:
                        edge_select_list.append(0)
                except IndexError:
                    print( pdb,'need to sort sugar first')
                    raise IndexError

            # bond between group

            interacting_list = []  #########################################
            for i, j, k in zip(bond_list, contact_list, CH_pi_list):
                interacting_list.append([i, j, k])

            clean_interacting_list = []  ###########################################
            row = []
            col = []
            for i, edge in enumerate(interacting_list):
                if edge != [0, 0, 0]:
                    # print(i,edge)
                    clean_interacting_list.append(edge)
                    # print(int(pairwise_matrix[0][i]),int(pairwise_matrix[1][i]))
                    row.append(int(pairwise_matrix[0][i]))
                    col.append(int(pairwise_matrix[1][i]))
            clean_pairwise_matrix = [row, col]  #####################################

            edge_select_interacting_list = []  ##############
            row = []
            col = []
            for i, select in enumerate(edge_select_list):
                if select == 1:
                    # print(i,edge)
                    edge_select_interacting_list.append(interacting_list[i])
                    # print(int(pairwise_matrix[0][i]),int(pairwise_matrix[1][i]))
                    row.append(int(pairwise_matrix[0][i]))
                    col.append(int(pairwise_matrix[1][i]))
            select_pairwise_matrix = [row, col]

            atom_resi_id_polar_interaction_count=[ct/2 for ct in atom_resi_id_polar_interaction_count ]
            atom_resi_id_inpolar_interaction_count = [ct / 2 for ct in atom_resi_id_inpolar_interaction_count]

            print('polar-polar/inpolar-inpolar non_covalent contact between residues within 3.5 A: ', sum(contact_list) / 2)
            print('\tpolar-polar:', sum(polar_contact_list) / 2)
            print('\tinpolar-inpolar:', sum(inpolar_contact_list) / 2)
            print('\tpolar interaction   :',atom_resi_id_polar_interaction_count)
            print('\tinpolar interaction :',atom_resi_id_inpolar_interaction_count)
            print('\tpolar interaction>1 :',[atom_resi_id_list[ct] for ct in range(len(atom_resi_id_list)) if atom_resi_id_polar_interaction_count[ct]>1])
            print('\tpolar interaction>2 :', [atom_resi_id_list[ct] for ct in range(len(atom_resi_id_list)) if
                                              atom_resi_id_polar_interaction_count[ct] > 2])
            print('polar-polar/inpolar-inpolar non_covalent contact between amino acid within 3.5 A: ',
                  sum(aa_contact_list) / 2)
            print('\tpolar-polar:', sum(polar_aa_contact_list) / 2)
            print('\tinpolar-inpolar:', sum(inpolar_aa_contact_list) / 2)

            print('to be predicted contact between pieces within 4 A: ', sum(edge_select_list) / 2)

            print(len(bond_list), sum(bond_list))
            print(len(contact_list), sum(contact_list))
            print(len(aa_contact_list), sum(aa_contact_list))
            print(len(edge_select_list), sum(edge_select_list))

            enablePrint()

        if 'bundle all info' != 1:
            info_dict={}
            if output_txt==1:
                if out_path != None:
                    out_path = out_path
                else:
                    out_path = loc

                with open(out_path + out_file_name, 'w') as out_file:
                    out_file.write('atom_resi_id_list\t' + str((atom_resi_id_list)) + '\n\n')
                    out_file.write('is_sugar_list\t' + str((is_sugar_list)) + '\n\n')
                    out_file.write('atom_coord_list\t' + str((atom_coord_list).tolist()) + '\n\n')
                    out_file.write('atomtype_id_list\t' + str(atomtype_id_list) + '\n\n')
                    out_file.write('polartype_list\t' + str(polartype_list) + '\n\n')
                    out_file.write('residue_id_list\t' + str(residue_id_list) + '\n\n')
                    out_file.write('atompiece_list\t' + str(atompiece_list) + '\n\n')
                    out_file.write('atompos_list\t' + str(atompos_list) + '\n\n')
                    out_file.write('atom_to_mole_id_list\t' + str(atom_to_mole_id_list) + '\n\n')
                    out_file.write('piece_list\t' + str(piece_list) + '\n\n')

                    out_file.write('atom_number\t' + str(len(atomtype_id_list)) + '\n\n')
                    out_file.write('CH_pi_info\t' + str(CH_pi_info) + '\n\n')

                    out_file.write('clean_pairwise_matrix\t' + str(clean_pairwise_matrix) + '\n\n')
                    out_file.write('clean_interacting_list\t' + str(clean_interacting_list) + '\n\n')
                    out_file.write('aa_contact_list\t' + str(aa_contact_list) + '\n\n')
                    out_file.write('edge_select\t' + str(edge_select.tolist()) + '\n\n')
                    out_file.write('edge_select_interacting_list\t' + str(edge_select_interacting_list) + '\n\n')
                    out_file.write('select_pairwise_matrix\t' + str(select_pairwise_matrix) + '\n\n')
                    out_file.write('polar interaction>1\t'+str([atom_resi_id_list[ct] for ct in range(len(atom_resi_id_list)) if
                                                      atom_resi_id_polar_interaction_count[ct] > 1])+ '\n\n')
                    out_file.write('polar interaction>2\t' + str([atom_resi_id_list[ct] for ct in range(len(atom_resi_id_list)) if
                                                          atom_resi_id_polar_interaction_count[ct] > 2]) + '\n\n')
                    out_file.close()
                return 'generated input'

            else:

                info_dict['atom_coord_list']=str((atom_coord_list).tolist())
                info_dict['atomtype_id_list']=str(atomtype_id_list)
                info_dict['polartype_list']=str(polartype_list)
                info_dict['residue_id_list']=str(residue_id_list)
                info_dict['atompiece_list']=str(atompiece_list)
                info_dict['atompos_list']=str(atompos_list)
                info_dict['piece_list']=str(piece_list)
                info_dict['atom_number'] = str(len(atomtype_id_list))

                info_dict['CH_pi_info']=str(CH_pi_info)

                info_dict['clean_pairwise_matrix']=str(clean_pairwise_matrix)
                info_dict['clean_interacting_list']=(clean_interacting_list)
                info_dict['aa_contact_list']=str(aa_contact_list)
                info_dict['edge_select']=str(edge_select.tolist())
                info_dict['edge_select_interacting_list']=str(edge_select_interacting_list)
                info_dict['select_pairwise_matrix']=str(select_pairwise_matrix)


            return info_dict
   #except NameError:
   #    raise
   except Exception as inst:
       raise
       enablePrint()
       print(inst)
       print(loc,pdb)
       blockPrint()
       return 'error\t'+pdb








def main():
    ### load CH-pi score

    loc='/home/tm21372/Rosetta/workspace/specificity_check/mut/all_pdb_pair/to_cst_relax/cst_fast/'
    loc='/home/tm21372/Rosetta/workspace/DL/sugar-binding-predictor/example/conor_struct_2/All_Rosetta_Output_Structures/'
    loc='/home/tm21372/Rosetta/workspace/sugar_protein_pair_dataset/3.5A_cutoff/relaxed_pair_structure/batch_1/cst_fast/'
    loc='/home/tm21372/Rosetta/workspace/sugar_protein_pair_dataset/ndv2_negative_extract/'
    #loc='/home/tm21372/Rosetta/workspace/poly_sac/manually_select/unrelax/'
    loc='/home/tm21372/Rosetta/workspace/3rd_design/design_after_hb/debug/'
    loc='/home/tm21372/Rosetta/workspace/sugar_protein_pair_dataset/3.5A_cutoff/relaxed_pair_structure/batch_1/cst_fast/test_new_ch-pi/'
    loc='/home/tm21372/Rosetta/workspace/DL/predictor_running/check/'
    loc='/home/tm21372/Rosetta/workspace/sugar_protein_pair_dataset/3.5A_cutoff/relaxed_pair_structure/batch_1/cst_fast/'
    out_loc='/home/tm21372/Rosetta/workspace/sugar_protein_pair_dataset/3.5A_cutoff/relaxed_pair_structure/batch_1/cst_fast/new_input/'
    loc='/home/tm21372/Rosetta/workspace/specificity_check/mut/all_pdb_pair_with_metal/docking/ligand_dock/'
    loc='/home/tm21372/Rosetta/workspace/CLIMBS_help_design/test/'
    loc='/home/tm21372/Rosetta/workspace/sugar_protein_pair_dataset/3.5A_cutoff/relaxed_pair_structure/batch_1/cst_fast/ndv6/di-sac-indicate/'
    #loc='/home/tm21372/Rosetta/workspace/CLIMBS_help_design/sugar_GLYCAM/mono-sac-unit/'
    loc='/home/tm21372/Rosetta/workspace/CLIMBS_help_docking/ndv6_chai1/mono_sac_unit/'
    file_list = []
    for f in os.listdir(loc):
        if re.search('(.*)\.pdb', f):
            if '_resi' not in f:
                if '_elicit' not in f:
                    #if 'di' not in f:
                        file_list.append(f)
    count_file = len(file_list)
    print(str(count_file) + ' files in under clearance')

    batch_size = 1
    max_len_seq = 0
    len_seq_record = []


    for pdb in tqdm(file_list[:]):#    24i-GAL_pred.rank_0.pdb            GAL.pdb ['24i_H3_pred.rank_0.pdb']:#
        try:
            #data_preprocessing(loc, pdb,debug_mode=1,ignore_CH_pi=0,generate_file_only=0,out_path=loc)

            ### process std mono-sac-unit naming for renaming
            data_preprocessing(loc, pdb, debug_mode = 0, ignore_CH_pi = 0, generate_file_only = 0, print_info = 1,save_new_sugar_example=1,save_exist_sugar=1,save_new_sugar_renaming=1,output_std_sugar_naming=1)
        except Exception:
            enablePrint()
            print(pdb)
            raise
    #data_preprocessing(loc, 'di3_cfr_int:A_B:3N95--H-GLC`1-`_-1g0p208a0i0s2o_pair_1_0001.pdb', debug_mode=1, ignore_CH_pi=0, generate_file_only=0, )
    #data_preprocessing('/home/tm21372/Rosetta/workspace/DL/attention_visualize/','cfr_int:A_B:1J0J--D-GLC`3-`_-2g0p9a0i0s0o_pair_2_0001.pdb', output_CH_pi_record=1, debug_mode=1)
    #data_preprocessing('/home/tm21372/Rosetta/workspace/3rd_design/previous_9_design/','TIM1.pdb')
    #data_preprocessing(loc, 'cfr_int:A_B:7VN6--M-GLC`1-`_-1g0p163a0i0s0o_pair_1_0001.pdb', debug_mode=1)
    #data_preprocessing('/home/tm21372/Rosetta/workspace/2nd_design/new_res_design/replace_by_sugar_all//', 'sort_116_W4B_4A1J_6.pdb', debug_mode=1,ignore_CH_pi=1,out_path='/home/tm21372/Rosetta/workspace/2nd_design/new_res_design/replace_by_sugar_all/GLC//train/')
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()


