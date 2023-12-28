import os,re,pymol,shutil,json,sys
from pymol import cmd,stored
import operator
import torch,math
import numpy as np
from measure_CH_pi import measure_CH_pi


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
def data_preprocessing(loc,pdb,output_txt=1,output_CH_pi_record=1,max_len_seq=0,batch_size=1):

    CH_pi_info_dict={}
    CH_pi_record=measure_CH_pi(loc, pdb,out_record=output_CH_pi_record)
    CH_pi_info_dict[pdb[:-4]]=CH_pi_record



    cmd.reinitialize()
    cmd.load(loc + pdb)
    cmd.remove('hydrogen')
    cmd.remove("(not (alt +'A') )and ( polymer)")
    cmd.remove('/////OXT')
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
    except KeyError:
        CH_pi_residue=''
    print('\tCH-pi residue',CH_pi_residue)


    cmd.select('atom_NOS','symbol O+N+S')
    cmd.select('polar_charged','(byres( ((not polymer) and '+pdb[:-4]+') around 3.5) and atom_NO) and (////ARG or ////HIS or ////LYS or ////ASP or ////GLU or ////Ser or ////THR or ////ASN or ////GLN or ////TYR)')



    if CH_pi_residue!='':
        cmd.select('aromatic',CH_pi_residue+' and ((not polymer) around 10)')
        take_chi_pi_record = '1'
    else:
        cmd.select('aromatic', 'None')
        take_chi_pi_record = '0'

    cmd.remove('not (not polymer or polar_charged or aromatic)')
    resi_num = cmd.select('ca', '/////CA')
    if resi_num == 0:
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
    if cmd.select('not polymer') == 0 or cmd.select('polymer') == 0:
        return 'No interacting fragment'
    cmd.save(loc + 'elicit.pdb')


    with open(loc + 'elicit.pdb', 'r') as pdb_file:
        context_line = pdb_file.readlines()
        pdb_file.close()
    os.remove(loc + 'elicit.pdb')

    #### generate input

    atom_line = [i for i in context_line if i[:4] == 'HETA' or i[:4] == 'ATOM']
    sugar_line = [i for i in context_line if i[:4] == 'HETA']
    if sugar_line == []:
        return 'No recognized sugar'
    print('atom number: ', len(atom_line))
    # print([[i[31:38].split()[0],i[39:45].split()[0],i[47:56].split()[0]] for i in atom_line])
    atom_coord_list = np.array([[i[31:38].split()[0], i[39:45].split()[0], i[47:56].split()[0]] for i in
                                atom_line])  ##################################################
    atom_coord_list = atom_coord_list.astype(float)
    resi_atom_name = [i[17:21].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]

    atomtype_list = [atom_type_dict[i] for i in resi_atom_name]  #########################################
    print(resi_atom_name)
    atomtype_id_list = [atomtype2id[atom_type_dict[i]] for i in resi_atom_name]  ##################
    print('atomtype involved: ', len(list(set(atomtype_list))))
    polartype_list = []  #############################
    for i in atomtype_list:
        if i[0] == 'N' or i[0] == 'O' or i[0] == 'S':
            polartype_list.append('polar')
        else:
            polartype_list.append('in-polar')
    residue_id_list = [i[17:21].split()[0] + '_' + i[22:30].split()[0] for i in atom_line]  ###########################
    print(residue_id_list)

    print('residues involved: ', len(list(set(residue_id_list))))

    atom_resi_id_list = [i[17:21].split()[0] + '_' + i[22:30].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]
    print(atom_resi_id_list)
    print((set(atom_resi_id_list)))
    if len(atom_resi_id_list) != len(set(atom_resi_id_list)):
        return 'duplicate atom in data-preprocessing, skip'

    atom_resi_id_list_for_sugar = [i[22:30].split()[0] + '_' + i[12:16].split()[0] for i in atom_line]
    # print(atom_resi_id_list_for_sugar)

    atompiece_list = []  ###### what type of fragment atom belong to
    for i in resi_atom_name:
        if i[:3] not in aa2fid:
            atompiece_list.append(sugar_atom2fragment[i][0])
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
        # print(ring_atom_id_list)
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

    bond_list = []  #######################
    contact_list = []  ##################
    CH_pi_list = []
    aa_contact_list = []  ###################
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
            # print(int(pairwise_matrix[0][i])+1,int(pairwise_matrix[1][i])+1)
            if (residue_id_list[pairwise_matrix[0][i]].split('_')[0] not in current_sugar_list) and (
                    residue_id_list[pairwise_matrix[1][i]].split('_')[0] not in current_sugar_list):
                aa_contact_list.append(1)
                # print(int(pairwise_matrix[0][i]) + 1, int(pairwise_matrix[1][i]) + 1)
            else:
                aa_contact_list.append(0)
        else:
            contact_list.append(0)
            aa_contact_list.append(0)

        #  if error here, sort sugar index before residue index
        # print(atompos_list[pairwise_matrix[1][i]],atompos_list[pairwise_matrix[0][i]])
        # print(atom_coord_list[pairwise_matrix[0][i]],atom_coord_list[pairwise_matrix[1][i]])
        if atompos_list[pairwise_matrix[0][i]] != atompos_list[pairwise_matrix[1][i]] and math.dist(
                atom_coord_list[pairwise_matrix[0][i]], atom_coord_list[pairwise_matrix[1][i]]) < 3.5:
            edge_select_list.append(1)
            edge_select[batch_size - 1][pairwise_matrix[0][i]][pairwise_matrix[1][i]] = 1
        else:
            edge_select_list.append(0)

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

    print('polar-polar/inpolar-inpolar non_covalent contact between residues within 3.5 A: ', sum(contact_list) / 2)
    print('polar-polar/inpolar-inpolar non_covalent contact between amino acid within 3.5 A: ',
          sum(aa_contact_list) / 2)
    print('to be predicted contact between pieces within 4 A: ', sum(edge_select_list) / 2)

    print(len(bond_list), sum(bond_list))
    print(len(contact_list), sum(contact_list))
    print(len(aa_contact_list), sum(aa_contact_list))
    print(len(edge_select_list), sum(edge_select_list))


    info_dict={}
    if output_txt==1:
        with open(loc + pdb[:-4] + '_elicit_info.txt', 'w') as out_file:
            out_file.write('atom_coord_list\t' + str((atom_coord_list).tolist()) + '\n\n')
            out_file.write('atomtype_id_list\t' + str(atomtype_id_list) + '\n\n')
            out_file.write('polartype_list\t' + str(polartype_list) + '\n\n')
            out_file.write('residue_id_list\t' + str(residue_id_list) + '\n\n')
            out_file.write('atompiece_list\t' + str(atompiece_list) + '\n\n')
            out_file.write('atompos_list\t' + str(atompos_list) + '\n\n')
            out_file.write('piece_list\t' + str(piece_list) + '\n\n')

            out_file.write('atom_number\t' + str(len(atomtype_id_list)) + '\n\n')
            out_file.write('CH_pi_info\t' + str(CH_pi_info) + '\n\n')

            out_file.write('clean_pairwise_matrix\t' + str(clean_pairwise_matrix) + '\n\n')
            out_file.write('clean_interacting_list\t' + str(clean_interacting_list) + '\n\n')
            out_file.write('aa_contact_list\t' + str(aa_contact_list) + '\n\n')
            out_file.write('edge_select\t' + str(edge_select.tolist()) + '\n\n')
            out_file.write('edge_select_interacting_list\t' + str(edge_select_interacting_list) + '\n\n')
            out_file.write('select_pairwise_matrix\t' + str(select_pairwise_matrix) + '\n\n')

            out_file.close()
    else:

        info_dict['atom_coord_list']=(atom_coord_list).tolist()
        info_dict['atomtype_id_list']=atomtype_id_list
        info_dict['polartype_list']=polartype_list
        info_dict['residue_id_list']=residue_id_list
        info_dict['atompiece_list']=atompiece_list
        info_dict['atompos_list']=atompos_list
        info_dict['piece_list']=piece_list

        info_dict['atom_number']=len(atomtype_id_list)
        info_dict['CH_pi_info']=CH_pi_info

        info_dict['clean_pairwise_matrix']=clean_pairwise_matrix
        info_dict['clean_interacting_list']=clean_interacting_list
        info_dict['aa_contact_list']=aa_contact_list
        info_dict['edge_select']=edge_select.tolist()
        info_dict['edge_select_interacting_list']=edge_select_interacting_list
        info_dict['select_pairwise_matrix']=select_pairwise_matrix
    return info_dict



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


atom_type_dict={}
for i in atom_type_line:
    try:
        atom_type_dict[i.split()[0]+'_'+i.split()[1]]=i.split()[2]
    except IndexError:
        pass

current_resi_atom_name_list=[]
for i in atom_type_dict:
    current_resi_atom_name_list.append(atom_type_dict[i])
current_atomtype_list=list(set(current_resi_atom_name_list))
#print('There are ',len(current_atomtype_list),' residue-atom names in  dict,')
#print('with ',len(list(set(current_atomtype_list))),' unique atomtypes in dict.\n')
unq_atomtype_list=(list(set(current_atomtype_list)))



atomtype2id={'CAbb': 1, 'CH0': 2, 'CH1': 3, 'CH2': 4, 'CH3': 5, 'CNH2': 6, 'COO': 7, 'CObb': 8, 'NH2O': 9, 'Narg': 10,
             'Nbb': 11, 'Nhis': 12, 'Nlys': 13, 'Npro': 14, 'NtrR': 15, 'Ntrp': 16, 'OCbb': 17, 'OH': 18, 'ONH2': 19, 'OOC': 20,
             'OS': 21, 'S': 22, 'SH1': 23, 'VIRT': 24, 'aroC': 25,'Phos':26,'SO4':27,}
#print(atomtype2id)

aa2fid={
    'SER':1, #Ser_Thr
    'THR':1, #Ser_Thr
    'ASN':2, #Asn_Gln
    'GLN':2, #Asn_Gln
    'CYS':3, #Cys_Met
    'PRO':4,
    'MET':3, #Cys_Met
    'PHE':5,
    'TYR':6,
    'TRP':7,
    'ARG':8,
    'HIS':9,
    'LYS':10,
    'ASP':11, #Asp_Glu
    'GLU':11, #Asp_Glu
    'GLY':12,  # aa backbone
}
'''
<s>=0,<pad>=13

'''
sugar_atom2fragment={
    'GLC_C1': [14,6], # 6-atom sugar ring
    'GLC_C2': [14,6],  # 6-atom sugar ring
    'GLC_C3': [14,6],  # 6-atom sugar ring
    'GLC_C4': [14,6],  # 6-atom sugar ring
    'GLC_C5': [14,6],  # 6-atom sugar ring
    'GLC_O5': [14,6],  # 6-atom sugar ring
    'GLC_O':  [15,1], # Oxygen
    'GLC_O1': [15,1],  # Oxygen
    'GLC_O2': [15,1],  # Oxygen
    'GLC_O3': [15,1],  # Oxygen
    'GLC_O4': [15,1],  # Oxygen
    'GLC_C6': [16,2],  # C-O
    'GLC_O6': [16,2],  # C-O

    'Glc_C1': [14, 6],  # 6-atom sugar ring
    'Glc_C2': [14, 6],  # 6-atom sugar ring
    'Glc_C3': [14, 6],  # 6-atom sugar ring
    'Glc_C4': [14, 6],  # 6-atom sugar ring
    'Glc_C5': [14, 6],  # 6-atom sugar ring
    'Glc_O5': [14, 6],  # 6-atom sugar ring
    'Glc_O': [15, 1],  # Oxygen
    'Glc_O1': [15, 1],  # Oxygen
    'Glc_O2': [15, 1],  # Oxygen
    'Glc_O3': [15, 1],  # Oxygen
    'Glc_O4': [15, 1],  # Oxygen
    'Glc_C6': [16, 2],  # C-O
    'Glc_O6': [16, 2],  # C-O

    'BGC_C1': [14, 6],  # 6-atom sugar ring
    'BGC_C2': [14, 6],  # 6-atom sugar ring
    'BGC_C3': [14, 6],  # 6-atom sugar ring
    'BGC_C4': [14, 6],  # 6-atom sugar ring
    'BGC_C5': [14, 6],  # 6-atom sugar ring
    'BGC_O5': [14, 6],  # 6-atom sugar ring
    'BGC_O': [15, 1],
    'BGC_O1': [15, 1],  # Oxygen
    'BGC_O2': [15, 1],  # Oxygen
    'BGC_O3': [15, 1],  # Oxygen
    'BGC_O4': [15, 1],  # Oxygen
    'BGC_C6': [16, 2],  # C-O
    'BGC_O6': [16, 2],  # C-O

    'GAL_C1': [14, 6],  # 6-atom sugar ring
    'GAL_C2': [14, 6],  # 6-atom sugar ring
    'GAL_C3': [14, 6],  # 6-atom sugar ring
    'GAL_C4': [14, 6],  # 6-atom sugar ring
    'GAL_C5': [14, 6],  # 6-atom sugar ring
    'GAL_O5': [14, 6],  # 6-atom sugar ring
    'GAL_O': [15, 1],
    'GAL_O1': [15, 1],  # Oxygen
    'GAL_O2': [15, 1],  # Oxygen
    'GAL_O3': [15, 1],  # Oxygen
    'GAL_O4': [15, 1],  # Oxygen
    'GAL_C6': [16, 2],  # C-O
    'GAL_O6': [16, 2],  # C-O

    'NAG_C1':[14, 6], # 6-atom sugar ring
    'NAG_C2':[14, 6], # 6-atom sugar ring
    'NAG_C3':[14, 6], # 6-atom sugar ring
    'NAG_C4':[14, 6], # 6-atom sugar ring
    'NAG_C5':[14, 6], # 6-atom sugar ring
    'NAG_O5':[14, 6], # 6-atom sugar ring
    'NAG_O3':[15, 1],  # Oxygen
    'NAG_O4':[15, 1],  # Oxygen
    'NAG_O1':[15, 1],  # Oxygen
    'NAG_O':[15, 1],  # Oxygen
    'NAG_C6':[16, 2], # C-O
    'NAG_O6':[16, 2], # C-O
    'NAG_C7':[17,4], # amide
    'NAG_C8':[17,4], # amide
    'NAG_N2':[17,4], # amide
    'NAG_O7':[17,4], # amide

    'BDP_C1' :[14, 6], # 6-atom sugar ring
    'BDP_C2' :[14, 6], # 6-atom sugar ring
    'BDP_C3' :[14, 6], # 6-atom sugar ring
    'BDP_C4' :[14, 6], # 6-atom sugar ring
    'BDP_O5': [14, 6],  # 6-atom sugar ring
    'BDP_C5' :[14, 6], # 6-atom sugar ring
    'BDP_O':[15, 1],  # Oxygen
    'BDP_O1':[15, 1],  # Oxygen
    'BDP_O2':[15, 1],  # Oxygen
    'BDP_O3':[15, 1],  # Oxygen
    'BDP_O4':[15, 1],  # Oxygen
    'BDP_C6' :[18,3], #COO
    'BDP_O6A':[18,3], #COO
    'BDP_O6B':[18,3], #COO

    'GCU_C1': [14, 6],  # 6-atom sugar ring
    'GCU_C2': [14, 6],  # 6-atom sugar ring
    'GCU_C3': [14, 6],  # 6-atom sugar ring
    'GCU_C4': [14, 6],  # 6-atom sugar ring
    'GCU_O5': [14, 6],  # 6-atom sugar ring
    'GCU_C5': [14, 6],  # 6-atom sugar ring
    'GCU_O': [15, 1],  # Oxygen
    'GCU_O1': [15, 1],  # Oxygen
    'GCU_O2': [15, 1],  # Oxygen
    'GCU_O3': [15, 1],  # Oxygen
    'GCU_O4': [15, 1],  # Oxygen
    'GCU_C6': [18, 3],  # COO
    'GCU_O6A': [18, 3],  # COO
    'GCU_O6B': [18, 3],  # COO

    'XYS_C1': [14, 6],  # 6-atom sugar ring
    'XYS_C2': [14, 6],  # 6-atom sugar ring
    'XYS_C3': [14, 6],  # 6-atom sugar ring
    'XYS_C4': [14, 6],  # 6-atom sugar ring
    'XYS_O5': [14, 6],  # 6-atom sugar ring
    'XYS_C5': [14, 6],  # 6-atom sugar ring
    'XYS_O': [15, 1],  # Oxygen
    'XYS_O1': [15, 1],  # Oxygen
    'XYS_O2': [15, 1],  # Oxygen
    'XYS_O3': [15, 1],  # Oxygen
    'XYS_O4': [15, 1],  # Oxygen

    'SIA_C2': [14, 6],  # 6-atom sugar ring
    'SIA_C3': [14, 6],  # 6-atom sugar ring
    'SIA_C4': [14, 6],  # 6-atom sugar ring
    'SIA_C5': [14, 6],  # 6-atom sugar ring
    'SIA_C6': [14, 6],  # 6-atom sugar ring
    'SIA_O6': [14, 6],  # 6-atom sugar ring
    'SIA_N5': [17, 4],  # amide
    'SIA_C10': [17, 4],  # amide
    'SIA_C11': [17, 4],  # amide
    'SIA_O10': [17, 4],  # amide
    'SIA_O1': [15, 1],  # Oxygen
    'SIA_O4': [15, 1],  # Oxygen
    'SIA_O': [15, 1],  # Oxygen
    'SIA_C1': [18, 3],  # COO
    'SIA_1O1': [18, 3],  # COO
    'SIA_2O1': [18, 3],  # COO
    'SIA_O7': [16, 2],  # C-O
    'SIA_C7': [16, 2],  # C-O
    'SIA_O8': [16, 2],  # C-O
    'SIA_C8': [16, 2],  # C-O
    'SIA_O9': [16, 2],  # C-O
    'SIA_C9': [16, 2],  # C-O

    'RIB_C1': [14, 5],  # 5-atom sugar ring
    'RIB_C2': [14, 5],  # 5-atom sugar ring
    'RIB_C3': [14, 5],  # 5-atom sugar ring
    'RIB_C4': [14, 5],  # 5-atom sugar ring
    'RIB_O4': [14, 5],  # 5-atom sugar ring
    'RIB_O': [15, 1],  # Oxygen
    'RIB_O1': [15, 1],  # Oxygen
    'RIB_O2': [15, 1],  # Oxygen
    'RIB_O3': [15, 1],  # Oxygen
    'RIB_C5': [16, 2],  # C-O
    'RIB_O5': [16, 2],  # C-O

    'Fru_C2': [14, 5],  # 5-atom sugar ring
    'Fru_C3': [14, 5],  # 5-atom sugar ring
    'Fru_C4': [14, 5],  # 5-atom sugar ring
    'Fru_C5': [14, 5],  # 5-atom sugar ring
    'Fru_O5': [14, 5],  # 5-atom sugar ring
    'Fru_O2': [15, 1],  # Oxygen
    'Fru_O3': [15, 1],  # Oxygen
    'Fru_O4': [15, 1],  # Oxygen
    'Fru_C1': [16, 2],  # C-O
    'Fru_O1': [16, 2],  # C-O
    'Fru_C6': [16, 2],  # C-O
    'Fru_O6': [16, 2],  # C-O

    'MAN_C1': [14, 6],  # 6-atom sugar ring
    'MAN_C2': [14, 6],  # 6-atom sugar ring
    'MAN_C3': [14, 6],  # 6-atom sugar ring
    'MAN_C4': [14, 6],  # 6-atom sugar ring
    'MAN_C5': [14, 6],  # 6-atom sugar ring
    'MAN_O5': [14, 6],  # 6-atom sugar ring
    'MAN_O': [15, 1],  # Oxygen
    'MAN_O1': [15, 1],  # Oxygen
    'MAN_O2': [15, 1],  # Oxygen
    'MAN_O3': [15, 1],  # Oxygen
    'MAN_O4': [15, 1],  # Oxygen
    'MAN_C6': [16, 2],  # C-O
    'MAN_O6': [16, 2],  # C-O

    'FUC_C1': [14, 6],  # 6-atom sugar ring
    'FUC_C2': [14, 6],  # 6-atom sugar ring
    'FUC_C3': [14, 6],  # 6-atom sugar ring
    'FUC_C4': [14, 6],  # 6-atom sugar ring
    'FUC_C5': [14, 6],  # 6-atom sugar ring
    'FUC_O5': [14, 6],  # 6-atom sugar ring
    'FUC_O': [15, 1],  # Oxygen
    'FUC_O1': [15, 1],  # Oxygen
    'FUC_O2': [15, 1],  # Oxygen
    'FUC_O3': [15, 1],  # Oxygen
    'FUC_O4': [15, 1],  # Oxygen
    'FUC_C6': [19, 1],  # Carbon

    'ARA_C1': [14, 6],  # 6-atom sugar ring
    'ARA_C2': [14, 6],  # 6-atom sugar ring
    'ARA_C3': [14, 6],  # 6-atom sugar ring
    'ARA_C4': [14, 6],  # 6-atom sugar ring
    'ARA_C5': [14, 6],  # 6-atom sugar ring
    'ARA_O5': [14, 6],  # 6-atom sugar ring
    'ARA_O': [15, 1],  # Oxygen
    'ARA_O1': [15, 1],  # Oxygen
    'ARA_O2': [15, 1],  # Oxygen
    'ARA_O3': [15, 1],  # Oxygen
    'ARA_O4': [15, 1],  # Oxygen

    'G6P_C1': [14, 6],  # 6-atom sugar ring
    'G6P_C2': [14, 6],  # 6-atom sugar ring
    'G6P_C3': [14, 6],  # 6-atom sugar ring
    'G6P_C4': [14, 6],  # 6-atom sugar ring
    'G6P_C5': [14, 6],  # 6-atom sugar ring
    'G6P_O5': [14, 6],  # 6-atom sugar ring
    'G6P_O': [15, 1],  # Oxygen
    'G6P_O1': [15, 1],  # Oxygen
    'G6P_O2': [15, 1],  # Oxygen
    'G6P_O3': [15, 1],  # Oxygen
    'G6P_O4': [15, 1],  # Oxygen
    'G6P_C6': [16, 2],  # C-O
    'G6P_O6': [16, 2],  # C-O
    'G6P_P': [20, 4],  # PO3
    'G6P_OP1': [20, 4],  # PO3
    'G6P_OP2': [20, 4],  # PO3
    'G6P_OP3': [20, 4],  # PO3
    'G6P_O1P': [20, 4],  # PO3
    'G6P_O2P': [20, 4],  # PO3
    'G6P_O3P': [20, 4],  # PO3

    'BG6_C1': [14, 6],  # 6-atom sugar ring
    'BG6_C2': [14, 6],  # 6-atom sugar ring
    'BG6_C3': [14, 6],  # 6-atom sugar ring
    'BG6_C4': [14, 6],  # 6-atom sugar ring
    'BG6_C5': [14, 6],  # 6-atom sugar ring
    'BG6_O5': [14, 6],  # 6-atom sugar ring
    'BG6_O': [15, 1],  # Oxygen
    'BG6_O1': [15, 1],  # Oxygen
    'BG6_O2': [15, 1],  # Oxygen
    'BG6_O3': [15, 1],  # Oxygen
    'BG6_O4': [15, 1],  # Oxygen
    'BG6_C6': [16, 2],  # C-O
    'BG6_O6': [16, 2],  # C-O
    'BG6_P': [20, 4],  # PO3
    'BG6_OP1': [20, 4],  # PO3
    'BG6_OP2': [20, 4],  # PO3
    'BG6_OP3': [20, 4],  # PO3
    'BG6_O1P': [20, 4],  # PO3
    'BG6_O2P': [20, 4],  # PO3
    'BG6_O3P': [20, 4],  # PO3

    'NGA_C1': [14, 6],  # 6-atom sugar ring
    'NGA_C2': [14, 6],  # 6-atom sugar ring
    'NGA_C3': [14, 6],  # 6-atom sugar ring
    'NGA_C4': [14, 6],  # 6-atom sugar ring
    'NGA_C5': [14, 6],  # 6-atom sugar ring
    'NGA_O5': [14, 6],  # 6-atom sugar ring
    'NGA_O3': [15, 1],  # Oxygen
    'NGA_O4': [15, 1],  # Oxygen
    'NGA_O1': [15, 1],  # Oxygen
    'NGA_O': [15, 1],  # Oxygen
    'NGA_C6': [16, 2],  # C-O
    'NGA_O6': [16, 2],  # C-O
    'NGA_C7': [17, 4],  # amide
    'NGA_C8': [17, 4],  # amide
    'NGA_N2': [17, 4],  # amide
    'NGA_O7': [17, 4],  # amide

    'A2G_C1': [14, 6],  # 6-atom sugar ring
    'A2G_C2': [14, 6],  # 6-atom sugar ring
    'A2G_C3': [14, 6],  # 6-atom sugar ring
    'A2G_C4': [14, 6],  # 6-atom sugar ring
    'A2G_C5': [14, 6],  # 6-atom sugar ring
    'A2G_O5': [14, 6],  # 6-atom sugar ring
    'A2G_O3': [15, 1],  # Oxygen
    'A2G_O4': [15, 1],  # Oxygen
    'A2G_O1': [15, 1],  # Oxygen
    'A2G_O': [15, 1],  # Oxygen
    'A2G_C6': [16, 2],  # C-O
    'A2G_O6': [16, 2],  # C-O
    'A2G_C7': [17, 4],  # amide
    'A2G_C8': [17, 4],  # amide
    'A2G_N2': [17, 4],  # amide
    'A2G_O7': [17, 4],  # amide

    'SGN_C1' : [14, 6],  # 6-atom sugar ring
    'SGN_C2' : [14, 6],  # 6-atom sugar ring
    'SGN_C3' : [14, 6],  # 6-atom sugar ring
    'SGN_C4' : [14, 6],  # 6-atom sugar ring
    'SGN_C5' : [14, 6],  # 6-atom sugar ring
    'SGN_C6' : [16, 2],  # C-O
    'SGN_N2' : [21, 1],  # Ntrp
    'SGN_O1' : [15, 1],  # Oxygen
    'SGN_O ' : [15, 1],  # Oxygen
    'SGN_O3' : [15, 1],  # Oxygen
    'SGN_O4' : [15, 1],  # Oxygen
    'SGN_O5' : [14, 6],  # 6-atom sugar ring
    'SGN_O6' : [16, 2],  # C-O
    'SGN_S1' : [22, 4],  # SO3
    'SGN_O1S': [22, 4],  # SO3
    'SGN_O2S': [22, 4],  # SO3
    'SGN_O3S': [22, 4],  # SO3
    'SGN_S2' : [22, 4],  # SO3
    'SGN_O4S': [22, 4],  # SO3
    'SGN_O5S': [22, 4],  # SO3
    'SGN_O6S': [22, 4],  # SO3

    'ASG_C1' : [14, 6],  # 6-atom sugar ring
    'ASG_C2' : [14, 6],  # 6-atom sugar ring
    'ASG_C3' : [14, 6],  # 6-atom sugar ring
    'ASG_C4' : [14, 6],  # 6-atom sugar ring
    'ASG_C5' : [14, 6],  # 6-atom sugar ring
    'ASG_C6' : [16, 2],  # C-O
    'ASG_C7' : [17, 4],  # amide
    'ASG_C8' : [17, 4],  # amide
    'ASG_N2' : [17, 4],  # amide
    'ASG_O' : [15, 1],  # Oxygen
    'ASG_O1' : [15, 1],  # Oxygen
    'ASG_O3' : [15, 1],  # Oxygen
    'ASG_O4' : [15, 1],  # Oxygen
    'ASG_O5' : [14, 6],  # 6-atom sugar ring
    'ASG_O6' : [16, 2],  # C-O
    'ASG_O7' : [17, 4],  # amide
    'ASG_S' : [22, 4],  # SO3
    'ASG_OSA' : [22, 4],  # SO3
    'ASG_OSB' : [22, 4],  # SO3
    'ASG_OSC' : [22, 4],  # SO3

    'RP5_C1': [14, 5],  # 5-atom sugar ring
    'RP5_C2': [14, 5],  # 5-atom sugar ring
    'RP5_C3': [14, 5],  # 5-atom sugar ring
    'RP5_C4': [14, 5],  # 5-atom sugar ring
    'RP5_O4': [14, 5],  # 5-atom sugar ring
    'RP5_O': [15, 1],  # Oxygen
    'RP5_O1': [15, 1],  # Oxygen
    'RP5_O2': [15, 1],  # Oxygen
    'RP5_O3': [15, 1],  # Oxygen
    'RP5_C5': [16, 2],  # C-O
    'RP5_O5': [16, 2],  # C-O
    "RP5_P'": [20, 4],  # PO3
    'RP5_O1X': [20, 4],  # PO3
    'RP5_O2X': [20, 4],  # PO3
    'RP5_O3X': [20, 4],  # PO3
}

current_sugar_list=['GLC','BGC','GAL','NAG','BDP','GCU','FUC','MAN','SIA','XYS','ARA','Fru','RIB','NGA','A2G','SGN','ASG','BG6','G6P','RP5','Glc']

aromatic_ring_atom_dict={
    'PHE':['CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
    'TYR':['CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
    'TRP':['NE1', 'CD1', 'CG', 'CD2', 'CE2'],
    'TRB':['CD2', 'CE2','CZ2','CH2','CZ3','CE3'],
}





def main():
    ### load CH-pi score

    loc = '../example/test_pdb/'
    loc = loc.replace('\\', '/')
    loc = loc + '/'



    file_list = []
    for f in os.listdir(loc):
        if re.search('(.*)\.pdb', f):
            file_list.append(f)
    count_file = len(file_list)
    print(str(count_file) + ' files in under clearance')

    batch_size = 1
    max_len_seq = 0
    len_seq_record = []

    for pdb in file_list:#['5XC2_51_poly_1.pdb']:#
        data_preprocessing(loc, pdb,)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

