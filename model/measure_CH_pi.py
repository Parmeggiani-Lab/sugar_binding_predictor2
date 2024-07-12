import ast
import shutil
import sys,os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

import os
import numpy as np
import math
from tqdm import tqdm
from pymol import cmd,stored

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/math.pi

def get_center(ring_coords_list):
    x_sum = 0
    y_sum = 0
    z_sum = 0
    count = 0
    for atom in ring_coords_list:
        x_sum += atom[0]
        y_sum += atom[1]
        z_sum += atom[2]
        count += 1

    vec1=ring_coords_list[0]-ring_coords_list[1]
    vec2 = ring_coords_list[0] - ring_coords_list[2]
    vec_vert=np.cross(vec1,vec2)
    return [x_sum / count, y_sum / count, z_sum / count],vec_vert

def motif_info_standardize(v1):
    return [(float(v1[0])-2)/(3.5-2),(float(v1[1])+180)/360,(float(v1[2])+180)/360,(float(v1[3])+180)/360]

def run_dist(v1, v2):
    try:
        dist= np.sqrt(np.sum((v1 - v2) ** 2))
    except TypeError:
        v1=np.array(v1)
        v2 = np.array(v2)
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist

def vec_pm_select(input_vec):
    output_list=[]
    for i,yes_no in enumerate(input_vec):
        if yes_no==1:
            output_list.append(str(i+1))
    #print(output_list)
    list_translated='+'.join(output_list)
    command='select target,i. '+list_translated
    print(command)
    return output_list

def angle(a0,a1,a2):
    vec1=[a1[0]-a0[0],a1[1]-a0[1],a1[2]-a0[2]]
    vec2 = [a2[0] - a0[0], a2[1] - a0[1], a2[0] - a0[2]]

    cos=(vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2])/(math.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2])*math.sqrt(vec2[0]*vec2[0]+vec2[1]*vec2[1]+vec2[2]*vec2[2]))
    angle=math.acos(cos)*180/math.pi
    if angle >360:
        angle=360-angle
    return angle

def dihedral(a0,a1,a3,a4):
    #vec0=[a1[0]-a0[0],a1[1]-a0[1],a1[2]-a0[2]]
    #vec1 = [a3[0] - a0[0], a3[1] - a0[1], a3[2] - a0[2]]
    #vec2 = [a4[0] - a1[0], a4[1] - a1[1], a4[2] - a1[2]]
    vec0=[a3[0]-a1[0],a3[1]-a1[1],a3[2]-a1[2]]
    vec1 = [a0[0] - a1[0], a0[1] - a1[1], a0[2] - a1[2]]
    vec2 = [a4[0] - a3[0], a4[1] - a3[1], a4[2] - a3[2]]

    vec_pl1=[vec1[1]*vec0[2]-vec1[2]*vec0[1],vec1[2]*vec0[0]-vec1[0]*vec0[2],vec1[0]*vec0[1]-vec1[1]*vec0[0]]
    vec_pl2 = [vec0[1] * vec2[2] - vec0[2] * vec2[1], vec0[2] * vec2[0] - vec0[0] * vec2[2], vec0[0] * vec2[1] - vec0[1] * vec2[0]]

    try:
        cos = (vec_pl1[0] * vec_pl2[0] + vec_pl1[1] * vec_pl2[1] + vec_pl1[2] * vec_pl2[2]) / (math.sqrt(vec_pl1[0] * vec_pl1[0] + vec_pl1[1] * vec_pl1[1] + vec_pl1[2] * vec_pl1[2]) * math.sqrt(vec_pl2[0] * vec_pl2[0] + vec_pl2[1] * vec_pl2[1] + vec_pl2[2] * vec_pl2[2]))
        sup_angle=math.acos(cos)*180/math.pi
        if sup_angle >360:
            sup_angle=sup_angle-360
    except ZeroDivisionError:
        sup_angle=180
    return 180-sup_angle

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



def find_CH_pi_aromatic_pymol(loc,each_file):
    cmd.reinitialize()
    cmd.load(loc+each_file)
    ###save ligand with H, and interacting aromatic ring
    cmd.select('ligand','not polymer')
    cmd.h_add('ligand')
    cmd.select('ligand', 'not polymer')
    cmd.select('aromatic_ring','(byres(ligand around 4.71) and (////HIS or ////TRP or ////PHE or ////TYR))')
    cmd.remove('not (ligand or aromatic_ring)')
    cmd.remove('hydrogen and aromatic_ring')

    cmd.remove('/////OH and ////TYR')
    cmd.select('aromatic_ring', '(byres(ligand around 4.71) and (////HIS or ////TRP or ////PHE or ////TYR)) ')
    cmd.remove('not (ligand or aromatic_ring)')
    cmd.zoom()
    # cmd.save(loc + each_file[:-4] + '.pse') ## debug

    ### load .pdb record
    cmd.set('pdb_conect_all', 'on')
    cmd.save(loc + each_file[:-4] + '_measuring.ent')
    with open(loc + each_file[:-4] + '_measuring.ent','r') as read_record:
        context_line=read_record.readlines()
        read_record.close()
    os.remove(loc + each_file[:-4] + '_measuring.ent')
    atom_line = [i for i in context_line if i[:4] == 'HETA' or i[:4] == 'ATOM']
    sugar_line = [i for i in context_line if i[:4] == 'HETA']
    if sugar_line == []:
        return 'No recognized sugar'
    link_line = [i for i in context_line if i[:4] == 'CONE']
    check_new_sugar = [
        [i[:4], i[7:11].split()[0], i[12:16].split()[0], i[17:21].split()[0], i[22:30].split()[0], i[77:78].split()[0],np.array([float(i[31:38].split()[0]), float(i[39:45].split()[0]), float(i[47:56].split()[0])])]
        for i in atom_line ]
    new_sugar_residx_atomidx = {}  ### new sugar id - atom id
    info_dict={}
    for i in check_new_sugar:
        new_sugar_residx_atomidx[i[1]] = i[4]
        info_dict[i[1]]=i
    #print(new_sugar_residx_atomidx)
    link_record_dict = {}
    for i in link_line:
        atom1 = i.split()[1]
        atom_tar = [atom for atom in i.split()[2:] if atom in new_sugar_residx_atomidx]
        if atom_tar != [] and atom1 in new_sugar_residx_atomidx:
            link_record_dict[atom1] = atom_tar

    loops_in_graph = find_unique_loops(link_record_dict)  ### find loop in linking record
    #print('loop:',loops_in_graph)

    ### generate C-H pair
    C_atom_on_sugar_loop=[i for i in list(set([atom for loop in loops_in_graph for atom in loop])) if info_dict[i][0]=='HETA' and info_dict[i][5]=='C']
    #print('C_atom_on_sugar_loop:', C_atom_on_sugar_loop)
    #print(check_new_sugar)
    #print(link_record_dict)

    CH_pair_list=[]
    for i in C_atom_on_sugar_loop:
        neigb_list=link_record_dict[i]#
        for neigb in neigb_list:
            if info_dict[neigb][5]=='H':
                #CH_pair_list.append([i,neigb])
                CH_pair_list.append(
                        [info_dict[i][4] + '_' + info_dict[i][2]+'_' + info_dict[neigb][2], info_dict[i][3],
                         info_dict[i][6], info_dict[i][6]-info_dict[neigb][6],
                         info_dict[neigb][6] ])
    #print('C-H pair',CH_pair_list)

    ### get aromatic ring
    aromatic_ring_list=[]
    aromatic_loop_list=[ i for i in loops_in_graph if list((set([info_dict[j][0] for j in i])))==['ATOM'] and len(i)<=6]
    for ring in aromatic_loop_list:
        center=get_center([(info_dict[i][6]) for i in ring])

        #print('dihedral_atom',dihedral_atom_coord)
        #print([info_dict[i][3] + '_' + info_dict[i][4] for i in ring], len(ring))
        if info_dict[ring[0]][3]=='TRP' and len(ring)==6:
            aromatic_ring_list.append([info_dict[ring[0]][4], 'TRP_B', center[0], center[1]])
        else:
            aromatic_ring_list.append([info_dict[ring[0]][4],info_dict[ring[0]][3], center[0], center[1]])



    ### get Chi-2 of aromatic ring
    dihedral_atom_list = ['N', 'CA', 'CB', 'CG']
    for ring_info in aromatic_ring_list:
        chi_measure_dict={}
        for atom in check_new_sugar:
            if atom[2] in dihedral_atom_list and atom[4]==ring_info[0]  :
                chi_measure_dict[atom[2]]=(atom[1])#(atom[2]+'_'+atom[3]+'_'+atom[4])
        #print(chi_measure_dict)
        chi2=dihedral(info_dict[chi_measure_dict['N']][6],info_dict[chi_measure_dict['CA']][6],info_dict[chi_measure_dict['CB']][6],info_dict[chi_measure_dict['CG']][6])
        #print('pymol_chi2',chi2)
        ring_info.append(str(chi2))

    #print('aromatic_loop', aromatic_ring_list)

    CH_pi_info_list = []
    for aromatic_ring in aromatic_ring_list:
        for CH_pair in CH_pair_list:
            chi2_ori = float(aromatic_ring[4])#float(pose.residue(int(aromatic_ring[0])).chi(2))
            chi2 = float("{:.2f}".format(chi2_ori))
            C_X_dist = run_dist(CH_pair[2], aromatic_ring[2])
            H_X_dist = run_dist(CH_pair[4], aromatic_ring[2])
            if H_X_dist < C_X_dist:
                CH_vert_angle = angle_between(CH_pair[3], aromatic_ring[3])
                if CH_vert_angle > 90:
                    CH_vert_angle = 180 - CH_vert_angle
                CX_vec = np.array(aromatic_ring[2]) - np.array(CH_pair[2])
                CX_vert_angle = angle_between(CX_vec, aromatic_ring[3])
                Cp_X_dist = C_X_dist * np.sin(CX_vert_angle * np.pi / 180)
                CH_pi_info_list.append(
                    [C_X_dist, CH_vert_angle, Cp_X_dist,aromatic_ring[0] , aromatic_ring[1],
                     CH_pair[0], chi2,
                     loc+each_file])
                # print(aromatic_ring[0],pose.pdb_info().number(aromatic_ring[0]))
    return CH_pi_info_list

def pose_CH_pi_score(CH_pi_aromatic_list):
    CH_pi_score=0
    CH_pi_record_list=[]
    for i in CH_pi_aromatic_list:
        #print(i)
        if i[0]<4.5: #C-X distance cutoff=4.5A
            if i[1]<40: #CH-Pi angle cutoff=40 degree
                if i[4]=='HIS' or i[4]=='TRP': #for His, Trp_A
                    if i[2]<1.6: #C-projection distance cutoff=1.6A
                        print(i)
                        CH_pi_score+=1
                        CH_pi_record_list.append(str(i[3])+'_'+i[4]+'_'+i[5]+'_'+str(i[6]))
                elif i[4]=='PHE' or i[4]=='TRP_B' or i[4]=='TYR': #for Phe, Trp_B, Tyr
                    if i[2]<2: #C-projection distance cutoff=2A
                        print(i)
                        CH_pi_score += 1
                        if i[4]=='TRP_B':
                            CH_pi_record_list.append(str(i[3]) + '_' + 'TRB' + '_' + i[5]+'_'+str(i[6]))
                        else:
                            CH_pi_record_list.append(str(i[3]) + '_' + i[4] + '_' + i[5] + '_' + str(i[6]))
    print('CH_pi_score :',CH_pi_score)
    return CH_pi_score,CH_pi_record_list

def measure_CH_pi(loc,each_file,out_record=1,out_path=None):
    #blockPrint()
    print('\n'+each_file)

    CH_pi_aromatic_list=find_CH_pi_aromatic_pymol(loc,each_file)
    #print('CH_pi_aromatic_list',len(CH_pi_aromatic_list),CH_pi_aromatic_list)
    result=pose_CH_pi_score(CH_pi_aromatic_list)

    CH_pi_score = result[0]
    CH_pi_record= result[1]

    if out_path!=None:
        out_path=out_path
    else:
        out_path=loc

    if out_record==1:
        f = open(out_path + 'CH_Pi_score.txt', 'a')
        f.write(str(CH_pi_score)+'\t'+str(CH_pi_record)+'\t'+each_file.split('/')[-1]+'\t'+each_file+'\n')
        f.close()
    #enablePrint()
    return CH_pi_record

def main():
    loc='../example/test_pdb/'
    loc='/home/tm21372/Rosetta/workspace/2nd_design/new_res_design/natural_rescore/relaxed_native/'
    loc='/home/tm21372/Rosetta/workspace/2nd_design/GLC/native/double_relax/'
    loc='/home/tm21372/Rosetta/workspace/2nd_design/GLC/native/min/'
    loc='/home/tm21372/Rosetta/workspace/DL/sugar-binding-predictor/example/CH_pi_update/'
    pdb_files = [file for file in os.listdir(loc) if file[-4:] == '.pdb' ]
    for pdb in tqdm(pdb_files):
        #print(pdb)
        measure_CH_pi(loc,pdb)

    #measure_CH_pi('/home/tm21372/Rosetta/workspace/2nd_design/GLC_control/','rel_2hph_0001.pdb')


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()